# Data Standards Quick Reference

## Field Naming Rules

| Context | Format | Example |
|---------|--------|---------|
| Python Models | snake_case | `created_at`, `file_size` |
| API Responses | camelCase | `createdAt`, `fileSize` |
| Database | snake_case | `created_at`, `file_size` |
| WebSocket Events | snake_case type, camelCase data | `type: "photo_uploaded"`, `data: { photoId: "123" }` |

## Common Field Mappings

| Backend (Python) | Frontend (TypeScript) | Notes |
|-----------------|----------------------|-------|
| `created_at` | `createdAt` | datetime → ISO string |
| `file_size` | `fileSize` | |
| `photo_id` | `photoId` | |
| `recipe_id` | `recipeId` | |
| `is_paused` | `isPaused` | |
| `operations` | `steps` | Recipe field |
| `page_size` | `pageSize` | |

## Status Values

Always use these exact values (lowercase):
- Photo: `pending`, `processing`, `completed`, `failed`, `rejected`
- Queue: `pending`, `processing`, `completed`, `failed`
- Priority: `low`, `normal`, `high`

## WebSocket Event Format

```json
{
  "type": "event_name_snake_case",
  "data": {
    "fieldNameCamelCase": "value"
  },
  "timestamp": "2024-01-07T12:00:00.000Z"
}
```

## API Response Transformation

All API responses are automatically transformed by `/api/middleware/transform.py`:
- ✅ snake_case → camelCase
- ✅ datetime → ISO string
- ✅ Recipe operations → steps
- ✅ Computed fields added (e.g., QueueStatus.total)

## Common Pitfalls

1. **Don't** mix naming conventions in the same context
2. **Don't** forget to transform nested objects
3. **Don't** use different status values than documented
4. **Do** use the transformation layer for all API responses
5. **Do** keep WebSocket event types in snake_case

## Quick Checks

Before committing:
1. Run TypeScript build: `npm run build`
2. Check for Python type errors
3. Verify API responses match frontend expectations
4. Test WebSocket events in browser console

## Need More Info?

- Full documentation: `DATA_STANDARDS.md`
- Recent changes: `DATA_STANDARDS_UPDATE_LOG.md`
- Implementation guide: `DATA_STANDARDS_IMPLEMENTATION.md`