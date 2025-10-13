# Contributing to MP4 ID Detector

Thank you for your interest in contributing to the MP4 ID Detector project! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/mp4-id-detector.git
cd mp4-id-detector

# Create and activate virtual environment
python -m venv cvenv
source cvenv/bin/activate  # On Windows: cvenv\Scripts\activate

# Install dependencies
pip install -r requirements_minimal.txt
pip install -r requirements.txt  # For development tools

# Verify installation
python install_dependencies.py
```

## 📝 How to Contribute

### 1. Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the upstream repository as a remote

```bash
git clone https://github.com/yourusername/mp4-id-detector.git
cd mp4-id-detector
git remote add upstream https://github.com/originalusername/mp4-id-detector.git
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Test Your Changes
```bash
# Run tests
pytest

# Check code style
black --check .
flake8 .

# Run the main scripts
python video_id_detector2_optimized.py
python crime_no_crime_zero_shot1.py --help
```

### 5. Commit Changes
```bash
git add .
git commit -m "Add: brief description of changes"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 🎯 Types of Contributions

### Bug Reports
- Use the GitHub issue tracker
- Provide detailed reproduction steps
- Include system information and error logs

### Feature Requests
- Describe the feature clearly
- Explain the use case and benefits
- Consider implementation complexity

### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation updates
- Test coverage improvements

## 📋 Coding Standards

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small

### Code Formatting
```bash
# Format code
black .

# Check formatting
black --check .

# Lint code
flake8 .
```

### Documentation
- Update README.md for user-facing changes
- Add docstrings to new functions
- Update comments for complex logic

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_detector.py
```

### Writing Tests
- Test new functionality thoroughly
- Include edge cases
- Mock external dependencies
- Ensure tests are deterministic

## 📦 Release Process

### Version Numbering
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in setup.py
- Create release notes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Tagged in git

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies
6. **Error Messages**: Full error traceback if applicable

## 💡 Feature Requests

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Additional Context**: Any other relevant information

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues (use private communication)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to MP4 ID Detector! 🎉
