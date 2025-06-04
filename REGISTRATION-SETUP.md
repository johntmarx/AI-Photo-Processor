# Immich Self-Registration Setup Complete! 🎉

## Access Points

### Registration Portal
- **Local**: http://YOUR-SERVER-IP/register  
- **External**: https://your-domain.com/register

### Immich Photos
- **Local**: http://YOUR-SERVER-IP  
- **External**: https://your-domain.com

## Registration Details

### Secret Key
Your registration secret key is: `[SET IN .env as REGISTRATION_SECRET]`

**Important**: This key is required for anyone to register. Share it only with trusted users!

### How Registration Works

1. Users visit `/register`
2. They fill out:
   - Full Name
   - Email Address  
   - Password (minimum 8 characters)
   - **Registration Key**: `[Provided by administrator]`
   - Storage Quota (50GB to 1TB options)
3. Upon successful registration, they're redirected to Immich login

## Security Features

✅ **Secret key is secure**:
- Never exposed in browser/client code
- Validated server-side only
- Hashed with bcrypt

✅ **Rate limiting**: 
- 5 registration attempts per IP per 15 minutes
- Prevents brute force attacks

✅ **Input validation**:
- Email format validation
- Password strength requirements
- Secure headers with Helmet.js

## Management

### View All Users
```bash
curl -s -X GET "http://YOUR-SERVER-IP:2283/api/admin/users" \
  -H "x-api-key: YOUR-IMMICH-API-KEY" | jq .
```

### Change Registration Key
Edit your `.env` file:
```
REGISTRATION_SECRET=your-new-secret-key
```
Then restart: `docker restart immich_registration`

### Monitor Registration Logs
```bash
docker logs -f immich_registration
```

## Managing Users
View current users using the API command above.

Your self-registration portal is now live and ready for users!