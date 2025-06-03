# Immich Self-Registration Service

A secure self-registration portal for Immich that requires users to know a secret key.

## Features

- 🔐 **Secure Registration**: Requires a secret key that's validated server-side (never exposed to browser)
- 🛡️ **Rate Limited**: Prevents brute force attempts (5 registrations per IP per 15 minutes)
- 📊 **Quota Management**: Administrators can set storage quotas for new users
- 🎨 **Clean UI**: Modern, responsive design that works on all devices
- 🔒 **Security Headers**: Implements helmet.js for security best practices

## Security Design

The secret key is:
- Never sent to the browser or exposed in client-side code
- Hashed using bcrypt on the server
- Validated server-side only
- Can be changed via environment variable

## Usage

1. Users visit `/register`
2. Fill out the registration form including:
   - Full name
   - Email address
   - Password (min 8 characters)
   - Registration key (provided by administrator)
   - Storage quota selection
3. Upon successful registration, users are redirected to Immich login

## Configuration

Set these in your `.env` file:

```env
REGISTRATION_SECRET=your-super-secret-key-here
IMMICH_API_KEY=your-admin-api-key
```

## Endpoints

- `GET /register` - Registration form
- `POST /api/register` - Submit registration
- `GET /health` - Health check endpoint