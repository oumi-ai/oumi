#!/bin/bash
set -e

# Allure Manager for Oumi E2E Tests
# This script helps manage Allure reports from cloud clusters

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLUSTER_PREFIX="oumi-${USER}-e2e-tests"
DEFAULT_CLOUD="gcp"
REPORT_DIR="./allure-reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    download CLUSTER_NAME    Download Allure results from cluster
    serve CLUSTER_NAME       Serve Allure report from cluster
    generate CLUSTER_NAME    Generate Allure report from cluster
    list                     List clusters with Allure results
    help                     Show this help message

Options:
    -c, --cloud CLOUD       Specify cloud provider (default: gcp)
    -p, --prefix PREFIX     Cluster name prefix
    -d, --report-dir DIR    Local directory for reports (default: ./allure-reports)

Examples:
    $0 download oumi-ryan-e2e-tests-a100-80gb-4-nonspot
    $0 serve oumi-ryan-e2e-tests-a100-80gb-4-nonspot
    $0 generate oumi-ryan-e2e-tests-a100-80gb-4-nonspot
    $0 list

EOF
}

download_allure_results() {
    local cluster_name=$1
    local report_dir=${2:-$REPORT_DIR}

    print_info "Downloading Allure results from cluster: $cluster_name"

    mkdir -p "$report_dir"

    # Download allure-results directory
    if rsync -avz --progress "${cluster_name}:/home/gcpuser/sky_workdir/allure-results/" "$report_dir/${cluster_name}/"; then
        print_success "Allure results downloaded to $report_dir/${cluster_name}/"
    else
        print_error "Failed to download Allure results from $cluster_name"
        return 1
    fi
}

serve_allure_report() {
    local cluster_name=$1
    local report_dir=${2:-$REPORT_DIR}
    local port=${3:-8080}

    print_info "Serving Allure report for cluster: $cluster_name"

    local results_dir="$report_dir/${cluster_name}"

    if [ ! -d "$results_dir" ]; then
        print_error "Allure results not found. Download first: $0 download $cluster_name"
        return 1
    fi

    print_info "Starting Allure server on port $port..."
    print_info "Open your browser to: http://localhost:$port"

    allure serve "$results_dir" -p "$port"
}

generate_allure_report() {
    local cluster_name=$1
    local report_dir=${2:-$REPORT_DIR}

    print_info "Generating Allure report for cluster: $cluster_name"

    local results_dir="$report_dir/${cluster_name}"
    local report_path="$report_dir/${cluster_name}-report"

    if [ ! -d "$results_dir" ]; then
        print_error "Allure results not found. Download first: $0 download $cluster_name"
        return 1
    fi

    allure generate "$results_dir" --clean -o "$report_path"
    print_success "Allure report generated at: $report_path"
    print_info "Open index.html in your browser to view the report"
}

list_clusters() {
    local cloud=${1:-$DEFAULT_CLOUD}
    local prefix=${2:-$CLUSTER_PREFIX}

    print_info "Listing clusters with Allure results on $cloud..."

    # Use sky status to get cluster names
    local clusters=$(sky status --cloud $cloud | grep "$prefix" | awk '{print $1}' || true)

    if [ -z "$clusters" ]; then
        print_warning "No clusters found with prefix: $prefix"
        return 0
    fi

    echo
    print_success "Found clusters:"
    echo "$clusters" | while read -r cluster; do
        if [ -n "$cluster" ]; then
            echo "  - $cluster"
            # Check if cluster has allure results
            if ssh "$cluster" "test -d /home/gcpuser/sky_workdir/allure-results" 2>/dev/null; then
                echo "    ✓ Has Allure results"
            else
                echo "    ✗ No Allure results"
            fi
        fi
    done
    echo
}

# Parse command line arguments
COMMAND=""
CLUSTER_NAME=""
CLOUD=$DEFAULT_CLOUD
PREFIX=$CLUSTER_PREFIX
REPORT_DIRECTORY=$REPORT_DIR

while [[ $# -gt 0 ]]; do
    case $1 in
        download|serve|generate|list|help)
            COMMAND="$1"
            shift
            ;;
        -c|--cloud)
            CLOUD="$2"
            shift 2
            ;;
        -p|--prefix)
            PREFIX="$2"
            shift 2
            ;;
        -d|--report-dir)
            REPORT_DIRECTORY="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                print_error "Unknown command: $1"
                show_usage
                exit 1
            elif [ -z "$CLUSTER_NAME" ]; then
                CLUSTER_NAME="$1"
            else
                print_error "Unexpected argument: $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Execute command
case $COMMAND in
    download)
        if [ -z "$CLUSTER_NAME" ]; then
            print_error "Cluster name is required"
            show_usage
            exit 1
        fi
        download_allure_results "$CLUSTER_NAME" "$REPORT_DIRECTORY"
        ;;
    serve)
        if [ -z "$CLUSTER_NAME" ]; then
            print_error "Cluster name is required"
            show_usage
            exit 1
        fi
        serve_allure_report "$CLUSTER_NAME" "$REPORT_DIRECTORY"
        ;;
    generate)
        if [ -z "$CLUSTER_NAME" ]; then
            print_error "Cluster name is required"
            show_usage
            exit 1
        fi
        generate_allure_report "$CLUSTER_NAME" "$REPORT_DIRECTORY"
        ;;
    list)
        list_clusters "$CLOUD" "$PREFIX"
        ;;
    help)
        show_usage
        ;;
    "")
        print_error "No command specified"
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
