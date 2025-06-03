# Immich API Documentation

Generated from OpenAPI specification v1.134.0

## Overview

The Immich API provides comprehensive access to all photo and video management functionality. This includes asset management, album operations, user management, search capabilities, and more.

**Base URL**: `http://your-immich-server:port/api`
**Authentication**: Most endpoints require authentication via API key or session token.

## API Categories (129 Total Endpoints)

### 🎭 Activities (3 endpoints)
- `GET /activities` - Get activities for an album
- `POST /activities` - Create a new activity
- `GET /activities/statistics` - Get activity statistics
- `DELETE /activities/{id}` - Delete an activity

### 👥 Admin (8 endpoints)
- `GET /admin/users` - Get all users (admin only)
- `POST /admin/users` - Create a new user (admin only)
- `DELETE /admin/users/{id}` - Delete a user (admin only)
- `GET /admin/users/{id}` - Get user details (admin only)
- `PUT /admin/users/{id}` - Update user (admin only)
- `POST /admin/users/{id}/restore` - Restore deleted user (admin only)
- `GET /admin/users/{id}/statistics` - Get user statistics (admin only)
- `POST /admin/notifications` - Send admin notifications

### 📁 Albums (6 endpoints)
- `GET /albums` - Get all albums
- `POST /albums` - Create a new album
- `GET /albums/statistics` - Get album statistics
- `GET /albums/{id}` - Get album details
- `PATCH /albums/{id}` - Update album
- `DELETE /albums/{id}` - Delete album
- `PUT /albums/{id}/assets` - Add assets to album
- `DELETE /albums/{id}/assets` - Remove assets from album

### 🔑 API Keys (3 endpoints)
- `GET /api-keys` - Get all API keys
- `POST /api-keys` - Create a new API key
- `GET /api-keys/{id}` - Get API key details
- `PUT /api-keys/{id}` - Update API key
- `DELETE /api-keys/{id}` - Delete API key

### 🖼️ Assets (12 endpoints)
- `GET /assets` - Search and get assets
- `POST /assets` - Upload new asset
- `PUT /assets` - Update asset
- `DELETE /assets` - Delete assets
- `POST /assets/bulk-upload-check` - Check bulk upload status
- `GET /assets/device/{deviceId}` - Get assets for device
- `POST /assets/exist` - Check if assets exist
- `POST /assets/jobs` - Run asset jobs
- `GET /assets/random` - Get random assets
- `GET /assets/statistics` - Get asset statistics
- `GET /assets/{id}` - Get asset details
- `PUT /assets/{id}` - Update asset details
- `GET /assets/{id}/original` - Get original asset file
- `GET /assets/{id}/thumbnail` - Get asset thumbnail
- `GET /assets/{id}/video/playback` - Get video for playback

### 🔐 Authentication (9 endpoints)
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout
- `GET /auth/status` - Get authentication status
- `POST /auth/change-password` - Change password
- `POST /auth/admin-sign-up` - Admin signup
- `POST /auth/validateToken` - Validate auth token
- `POST /auth/pin-code` - Set PIN code
- `PUT /auth/pin-code` - Update PIN code
- `DELETE /auth/pin-code` - Remove PIN code

### 📥 Download (2 endpoints)
- `POST /download/archive` - Download assets as archive
- `POST /download/info` - Get download info

### 🔍 Search (8 endpoints)
- `GET /search/cities` - Search cities
- `GET /search/explore` - Explore search
- `POST /search/metadata` - Search by metadata
- `GET /search/person` - Search people
- `GET /search/places` - Search places
- `POST /search/random` - Random search
- `POST /search/smart` - Smart search
- `GET /search/suggestions` - Get search suggestions

### 🏠 Server (13+ endpoints)
- `GET /server/ping` - Health check
- `GET /server/about` - Server information
- `GET /server/config` - Server configuration
- `GET /server/features` - Available features
- `GET /server/statistics` - Server statistics
- `GET /server/storage` - Storage information
- `GET /server/theme` - Theme settings
- `GET /server/media-types` - Supported media types

### 👤 Users (7 endpoints)
- `GET /users` - Get all users
- `GET /users/me` - Get current user
- `PUT /users/me` - Update current user
- `GET /users/me/preferences` - Get user preferences
- `PUT /users/me/preferences` - Update user preferences
- `POST /users/profile-image` - Upload profile image
- `DELETE /users/profile-image` - Delete profile image

### 📚 Libraries (5 endpoints)
- `GET /libraries` - Get all libraries
- `POST /libraries` - Create library
- `GET /libraries/{id}` - Get library details
- `PUT /libraries/{id}` - Update library
- `DELETE /libraries/{id}` - Delete library
- `POST /libraries/{id}/scan` - Scan library

### 🗺️ Map (2 endpoints)
- `GET /map/markers` - Get map markers
- `GET /map/reverse-geocode` - Reverse geocode location

### 🧑‍🤝‍🧑 People (6 endpoints)
- `GET /people` - Get all people
- `POST /people` - Create person
- `PUT /people` - Update people
- `GET /people/{id}` - Get person details
- `PUT /people/{id}` - Update person
- `POST /people/{id}/merge` - Merge people

### 🏷️ Tags (4 endpoints)
- `GET /tags` - Get all tags
- `POST /tags` - Create tag
- `PUT /tags` - Update tags
- `GET /tags/{id}` - Get tag details
- `PUT /tags/{id}` - Update tag
- `DELETE /tags/{id}` - Delete tag

## Common Usage Examples

### Upload an Asset
```bash
curl -X POST "http://your-server:2283/api/assets" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "assetData=@photo.jpg" \
  -F "deviceAssetId=unique-id" \
  -F "deviceId=your-device" \
  -F "fileCreatedAt=2024-01-01T00:00:00.000Z" \
  -F "fileModifiedAt=2024-01-01T00:00:00.000Z"
```

### Get All Albums
```bash
curl -X GET "http://your-server:2283/api/albums" \
  -H "x-api-key: YOUR_API_KEY"
```

### Search Assets
```bash
curl -X POST "http://your-server:2283/api/search/smart" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "beach sunset", "type": "SMART_SEARCH"}'
```

### Get Server Info
```bash
curl -X GET "http://your-server:2283/api/server/ping"
```

## Authentication Methods

1. **API Key**: Add `x-api-key: YOUR_KEY` header
2. **Session Token**: Use cookie-based authentication after login
3. **Bearer Token**: Add `Authorization: Bearer TOKEN` header

## Important Notes

- Replace `your-server:2283` with your actual Immich server URL
- Most endpoints require authentication
- The API follows RESTful conventions
- Responses are in JSON format
- File uploads use multipart/form-data

## Generated Files

This documentation was generated using the Immich development environment. The complete OpenAPI specification is available in:
- `immich-openapi-specs.json` - Complete OpenAPI 3.0 specification

For the most up-to-date API documentation, refer to the official Immich documentation at https://immich.app/docs/