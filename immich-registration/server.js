const express = require('express');
const axios = require('axios');
const bcrypt = require('bcrypt');
const path = require('path');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Trust proxy since we're behind nginx
app.set('trust proxy', 1);

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "https://challenges.cloudflare.com", "https://*.cloudflare.com"],
      imgSrc: ["'self'", "data:"],
      objectSrc: ["'none'"],
      baseUri: ["'self'"],
      frameSrc: ["'self'", "https://challenges.cloudflare.com", "https://*.cloudflare.com"],
      connectSrc: ["'self'", "https://*.cloudflare.com"],
    },
  },
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Limit each IP to 5 requests per windowMs
  message: 'Too many registration attempts, please try again later.'
});

app.use('/api/register', limiter);

// Hash the secret key on startup for comparison
const SECRET_KEY_HASH = bcrypt.hashSync(process.env.REGISTRATION_SECRET || 'changeme', 10);

// Serve static files
app.use(express.static('public'));

// Registration endpoint
app.post('/api/register', async (req, res) => {
  const { name, email, password, secretKey, quotaGB } = req.body;

  // Validate inputs
  if (!name || !email || !password || !secretKey) {
    return res.status(400).json({ error: 'All fields are required' });
  }

  // Validate email format
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ error: 'Invalid email format' });
  }

  // Validate password strength
  if (password.length < 8) {
    return res.status(400).json({ error: 'Password must be at least 8 characters long' });
  }

  // Verify secret key
  const isValidSecret = bcrypt.compareSync(secretKey, SECRET_KEY_HASH);
  if (!isValidSecret) {
    return res.status(403).json({ error: 'Invalid registration key' });
  }

  try {
    // Calculate quota in bytes (default 100GB)
    const quotaSizeInBytes = (quotaGB || 100) * 1024 * 1024 * 1024;

    // Create user via Immich API
    // Generate unique storage label to avoid constraint violation
    const storageLabel = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const response = await axios.post(
      `${process.env.IMMICH_API_URL}/api/admin/users`,
      {
        email,
        name,
        password,
        quotaSizeInBytes,
        storageLabel,
        shouldChangePassword: false
      },
      {
        headers: {
          'x-api-key': process.env.IMMICH_API_KEY,
          'Content-Type': 'application/json'
        }
      }
    );

    // Log successful registration
    console.log(`New user registered: ${email}`);

    res.json({ 
      success: true, 
      message: 'Registration successful! You can now log in to Immich.',
      userId: response.data.id
    });

  } catch (error) {
    console.error('Registration error:', error.response?.data || error.message);
    
    // Handle specific Immich errors
    if (error.response?.status === 409) {
      return res.status(409).json({ error: 'Email already registered' });
    }
    
    res.status(500).json({ error: 'Registration failed. Please try again later.' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

app.listen(PORT, () => {
  console.log(`Immich Registration Service running on port ${PORT}`);
  console.log(`Immich API URL: ${process.env.IMMICH_API_URL}`);
  console.log(`Secret key is ${process.env.REGISTRATION_SECRET ? 'configured' : 'using default (CHANGE THIS!)'}`);
});