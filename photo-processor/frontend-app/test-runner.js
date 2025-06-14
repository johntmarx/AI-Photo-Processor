#!/usr/bin/env node

// Simple test runner to avoid vitest configuration issues
const { execSync } = require('child_process');
const path = require('path');

// Set working directory to frontend
process.chdir(__dirname);

const testFiles = [
  'src/lib/__tests__/utils.test.ts',
  'src/components/ui/__tests__/Button.test.tsx',
  'src/components/dashboard/__tests__/StatsCard.test.tsx',
  'src/services/__tests__/api.test.ts'
];

let passedTests = 0;
let totalTests = 0;

console.log('🧪 Running Frontend Tests...\n');

testFiles.forEach(testFile => {
  try {
    console.log(`Running: ${testFile}`);
    const result = execSync(`npx vitest run ${testFile} --reporter=basic`, { 
      encoding: 'utf8',
      timeout: 30000
    });
    
    // Count passed tests from output
    const matches = result.match(/✓.*?(\d+) tests?\)/g);
    if (matches) {
      const passed = matches.length;
      passedTests += passed;
      totalTests += passed;
      console.log(`✅ ${passed} tests passed\n`);
    }
  } catch (error) {
    console.log(`❌ Tests failed in ${testFile}`);
    console.log(error.stdout || error.message);
    totalTests += 1; // Assume at least 1 test failed
    console.log('');
  }
});

console.log(`\n📊 Final Results: ${passedTests}/${totalTests} tests passing`);
console.log(`Success Rate: ${((passedTests/totalTests) * 100).toFixed(1)}%`);

if (passedTests === totalTests) {
  console.log('🎉 All tests passed!');
  process.exit(0);
} else {
  console.log('⚠️  Some tests failed');
  process.exit(1);
}