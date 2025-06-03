# Immich Self-Registration Setup Complete! 🎉

## Access Points

### Registration Portal
- **Local**: http://192.168.1.114/register  
- **External**: https://photos.marxfamily.net/register

### Immich Photos
- **Local**: http://192.168.1.114  
- **External**: https://photos.marxfamily.net

## Registration Details

### Secret Key
Your registration secret key is: `SummerField-TigerSharks-Photography-2025`

**Important**: This key is required for anyone to register. Share it only with trusted users!

### How Registration Works

1. Users visit `/register`
2. They fill out:
   - Full Name
   - Email Address  
   - Password (minimum 8 characters)
   - **Registration Key**: `SummerField-TigerSharks-Photography-2025`
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
curl -s -X GET "http://192.168.1.114:2283/api/admin/users" \
  -H "x-api-key: RvgMnUnKXrC0xuxWoUrsKcPJwujjbLFuIZeIO6w3M" | jq .
```

### Change Registration Key
Edit `/home/john/immich/.env`:
```
REGISTRATION_SECRET=your-new-secret-key
```
Then restart: `docker restart immich_registration`

### Monitor Registration Logs
```bash
docker logs -f immich_registration
```

## Current Users
1. **john.t.marx@gmail.com** (Admin)
2. **john@john.com** (500GB quota)
3. **test@example.com** (Test user)

Your self-registration portal is now live and ready for users!