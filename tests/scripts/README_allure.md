# Allure Reports for Oumi E2E Tests

This directory contains the Allure Reports setup for Oumi E2E tests. Allure provides beautiful, interactive test reports with detailed logs, screenshots, and performance metrics.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your environment
conda activate oumi

# Install Allure pytest plugin
pip install allure-pytest

# Install Allure command-line tool (macOS)
brew install allure
```

### 2. Launch Tests with Allure

```bash
# Launch E2E tests with Allure reporting
./tests/scripts/launch_tests_allure.sh
```

### 3. View Results

```bash
# List available clusters
./tests/scripts/allure_manager.sh list

# Download results from a specific cluster
./tests/scripts/allure_manager.sh download <cluster-name>

# Serve the report locally
./tests/scripts/allure_manager.sh serve <cluster-name>

# Or generate a static report
./tests/scripts/allure_manager.sh generate <cluster-name>
```

## 📁 Files

- `e2e_tests_job_allure.yaml` - Allure-enhanced test configuration
- `allure_manager.sh` - Script to manage Allure reports
- `launch_tests_allure.sh` - Script to launch tests with Allure
- `README_allure.md` - This documentation

## 🎯 Features

### Beautiful Test Reports

- **Test execution timeline** - See when tests started and finished
- **Pass/fail statistics** - Visual breakdown of test results
- **Screenshots and logs** - Attachments for failed tests
- **Performance metrics** - Test duration and resource usage
- **Error details** - Stack traces and error messages

### Team Collaboration

- **Share reports via URL** - Serve reports locally or on a server
- **Historical trends** - Track test performance over time
- **Issue tracking integration** - Link tests to GitHub issues
- **Custom dashboards** - Create team-specific views

### Log Management

- **Structured log viewing** - Organized by test and step
- **Error highlighting** - Easy to spot failures
- **Search and filter** - Find specific tests or errors
- **Export capabilities** - Download reports for offline viewing

## 🔧 Usage

### Launch Tests

```bash
# Basic launch
./tests/scripts/launch_tests_allure.sh

# With custom options
export E2E_CLUSTER="aws"  # Use AWS instead of GCP
export E2E_USE_SPOT_VM=1   # Use spot instances
./tests/scripts/launch_tests_allure.sh
```

### Manage Reports

```bash
# List all clusters
./tests/scripts/allure_manager.sh list

# Download results from a specific cluster
./tests/scripts/allure_manager.sh download oumi-ryan-e2e-tests-allure-a100-80gb-4-nonspot

# Serve report on port 8080
./tests/scripts/allure_manager.sh serve oumi-ryan-e2e-tests-allure-a100-80gb-4-nonspot

# Generate static report
./tests/scripts/allure_manager.sh generate oumi-ryan-e2e-tests-allure-a100-80gb-4-nonspot

# Custom port and directory
./tests/scripts/allure_manager.sh serve oumi-ryan-e2e-tests-allure-a100-80gb-4-nonspot -p 9000 -d ./my-reports
```

### View Reports

1. **Interactive Server**: `./tests/scripts/allure_manager.sh serve <cluster>`
   - Opens browser automatically
   - Real-time updates
   - Best for development

2. **Static Report**: `./tests/scripts/allure_manager.sh generate <cluster>`
   - Creates HTML files
   - Can be shared via web server
   - Good for CI/CD integration

## 📊 Report Features

### Test Overview

- **Summary statistics** - Total tests, passed, failed, skipped
- **Execution timeline** - When tests ran and duration
- **Environment info** - GPU type, cloud provider, etc.

### Detailed Test Results

- **Test steps** - Individual test actions and assertions
- **Attachments** - Screenshots, logs, error details
- **Performance data** - Memory usage, execution time
- **Error context** - Stack traces, error messages

### Team Features

- **Issue linking** - Connect tests to GitHub issues
- **Test categorization** - Group by feature, priority, etc.
- **Historical trends** - Track performance over time
- **Custom dashboards** - Create team-specific views

## 🔄 Integration with Existing Workflow

### Replace Custom Dashboard

The Allure setup replaces the custom Flask dashboard with:

- ✅ **Industry standard** - Used by thousands of teams
- ✅ **Rich features** - Screenshots, logs, metrics, trends
- ✅ **Team ready** - Built for collaboration
- ✅ **Extensible** - Plugins for CI/CD integration
- ✅ **Maintained** - Active development and support

### Migration Steps

1. **Install Allure**: `pip install allure-pytest && brew install allure`
2. **Use new config**: `e2e_tests_job_allure.yaml` instead of `e2e_tests_job.yaml`
3. **Launch with Allure**: `./tests/scripts/launch_tests_allure.sh`
4. **View results**: Use `allure_manager.sh` instead of custom dashboard

## 🛠️ Troubleshooting

### Common Issues

**Allure not found**

```bash
# Install Allure command-line tool
brew install allure

# Or download manually from:
# https://github.com/allure-framework/allure2/releases
```

**No results found**

```bash
# Check if cluster has allure results
ssh <cluster-name> "ls -la /home/gcpuser/sky_workdir/allure-results/"

# Download results first
./tests/scripts/allure_manager.sh download <cluster-name>
```

**Port already in use**

```bash
# Use different port
./tests/scripts/allure_manager.sh serve <cluster-name> -p 9000
```

**Cluster not found**

```bash
# List available clusters
./tests/scripts/allure_manager.sh list

# Check sky status
sky status
```

## 📈 Benefits Over Custom Dashboard

| Feature | Custom Dashboard | Allure Reports |
|---------|------------------|----------------|
| **Installation** | Complex setup | Simple `pip install` |
| **Dependencies** | Flask, SQLite, custom code | Industry standard |
| **Features** | Basic logs and status | Rich reports, screenshots, trends |
| **Team Sharing** | Manual setup | Built-in sharing |
| **Maintenance** | Custom code to maintain | Active community |
| **Integration** | Manual CI/CD setup | Plugins available |
| **Documentation** | Custom docs | Extensive docs |

## 🎉 Next Steps

1. **Launch your first Allure test**:

   ```bash
   ./tests/scripts/launch_tests_allure.sh
   ```

2. **View the beautiful reports**:

   ```bash
   ./tests/scripts/allure_manager.sh serve <cluster-name>
   ```

3. **Share with your team**:
   - Generate static reports
   - Host on web server
   - Integrate with CI/CD

4. **Customize for your needs**:
   - Add custom test categories
   - Configure issue tracking
   - Set up team dashboards

Enjoy your beautiful test reports! 🎊
