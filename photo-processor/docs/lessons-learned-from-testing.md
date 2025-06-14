# Lessons Learned from Phase 0 Testing

## Critical Insights for Future Development

### 1. API Contract Consistency is Crucial
**Issue**: HashTracker had `is_processed()` in one version but `is_already_processed()` in another.

**Lesson**: 
- Always verify method names match between components before integration
- Consider creating interface definitions or abstract base classes
- Document the expected API in component docstrings

**Action Item**: Create an `interfaces.py` file defining expected methods for each component

### 2. Mock Testing with Image Processing Libraries
**Issue**: cv2.warpAffine failed when passed a MagicMock instead of a numpy array

**Lesson**:
- Image processing libraries expect specific data types (numpy arrays)
- Mock objects must match the expected data structure, not just the interface
- Consider creating reusable test fixtures for image data

**Best Practice**:
```python
# Good - returns actual numpy array
mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

# Bad - returns MagicMock
mock_cv2.imread.return_value = MagicMock()
```

### 3. Schema Validation Strictness
**Issue**: ColorAnalysis expected exact literal values and integer types

**Lessons**:
- Pydantic schemas are strict about types (int vs float)
- Literal types must match exactly ("normal" not "good")
- Type conversion should happen at the boundary, not deep in the code

**Recommendation**: Add conversion utilities for common transformations

### 4. Import-Time vs Runtime Dependencies
**Issue**: cv2 imported inside methods made mocking complex

**Lesson**:
- Consider the testability impact of where imports are placed
- Runtime imports can be harder to mock but reduce startup dependencies
- Document testing strategies for components with heavy dependencies

### 5. Component Integration Points
**Discovery**: The image processor has specific methods we weren't using correctly

**Key Methods Identified**:
- `convert_raw_to_rgb()` - for RAW file processing
- `save_high_quality_jpeg()` - for saving processed images
- `enhance_image()` - expects ColorAnalysis object
- No `load_image()` or `save_image()` methods exist

**Action**: Always read the actual component code, not just assume method names

### 6. File System Operations in Tests
**Success**: Using tempfile and proper cleanup worked perfectly

**Best Practice**:
```python
@pytest.fixture
def temp_dirs(self):
    base_dir = tempfile.mkdtemp()
    # ... create subdirs
    yield dirs
    shutil.rmtree(base_dir)  # Automatic cleanup
```

### 7. Original File Preservation Pattern
**Critical Success**: Copy-then-process pattern prevents data loss

**Pattern**:
1. Copy to permanent storage first
2. Process the copy
3. Only delete inbox file after successful upload
4. Keep original even after processing

This should be the standard pattern for any destructive operations.

### 8. Error Handling Granularity
**Observation**: Individual file failures shouldn't stop the batch

**Pattern Learned**:
- Process files individually with try/catch
- Log failures but continue processing
- Return results showing what succeeded/failed
- Never lose data due to processing errors

## Recommendations for Future Phases

### 1. Create Component Interfaces
Define clear contracts for all major components:
```python
# interfaces.py
class IHashTracker(Protocol):
    def is_already_processed(self, filepath: str) -> bool: ...
    def mark_as_processed(self, filepath: str) -> None: ...
```

### 2. Standardize Test Fixtures
Create a `test_fixtures.py` with common test data:
- Sample numpy arrays for images
- Valid ColorAnalysis objects
- Mock recipe objects
- Test file paths

### 3. Integration Test Harness
Before deploying:
- Test with actual Immich instance
- Verify cv2 operations with real images
- Test with various RAW formats
- Measure performance with large batches

### 4. Documentation Standards
For each component, document:
- Available methods and their signatures
- Expected input/output types
- Error conditions
- Example usage
- Testing approach

### 5. Continuous Validation
- Add pre-commit hooks to run tests
- Consider property-based testing for recipe operations
- Add integration tests to CI/CD pipeline
- Monitor production for edge cases

## Code Smells to Avoid

1. **Assuming Method Names**: Always verify the actual API
2. **Weak Mocks**: Ensure mocks match the real data structure
3. **Type Mismatches**: Be explicit about int vs float, especially with schemas
4. **Hidden Dependencies**: Make dependencies visible and mockable
5. **Data Loss Risks**: Always preserve originals before processing

## Success Patterns to Replicate

1. **Modular Design**: Each component has a single responsibility
2. **Clear Data Flow**: Original → Copy → Process → Upload
3. **Comprehensive Testing**: Unit tests caught issues before deployment
4. **Fail-Safe Defaults**: Preserve originals is always true
5. **Audit Trail**: Recipe system tracks all operations

These lessons will help ensure Phase 1 and beyond maintain the same quality and reliability standards established in Phase 0.