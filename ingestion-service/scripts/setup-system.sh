#!/bin/bash
set -e

echo "🛠️  Wessley.ai System Setup"
echo "============================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
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

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    print_error "Unsupported OS: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Check for system dependencies
print_status "Checking system dependencies..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    echo "Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
else
    print_success "Docker is installed"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose not found, checking for 'docker compose' plugin..."
    if ! docker compose version &> /dev/null; then
        print_error "Neither docker-compose nor 'docker compose' plugin is available"
        echo "Install Docker Compose from: https://docs.docker.com/compose/install/"
        exit 1
    else
        print_warning "Using 'docker compose' plugin instead of docker-compose"
        # Create alias for the rest of the script
        alias docker-compose='docker compose'
    fi
else
    print_success "Docker Compose is installed"
fi

# Check Python 3.11+
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$PYTHON_VERSION >= 3.11" | bc -l) )); then
        PYTHON_CMD="python3"
    else
        print_error "Python 3.11+ is required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    print_error "Python 3.11+ is not installed"
    if [[ "$OS" == "macos" ]]; then
        echo "Install with: brew install python@3.11"
    elif [[ "$OS" == "linux" ]]; then
        echo "Install with: sudo apt-get install python3.11 python3.11-pip"
    fi
    exit 1
fi
print_success "Python $($PYTHON_CMD --version) is available"

# Check Poetry
if ! command -v poetry &> /dev/null; then
    print_warning "Poetry is not installed"
    read -p "Install Poetry now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
        export PATH="$HOME/.local/bin:$PATH"
        if command -v poetry &> /dev/null; then
            print_success "Poetry installed successfully"
        else
            print_error "Poetry installation failed"
            exit 1
        fi
    else
        print_error "Poetry is required. Install with: curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
else
    print_success "Poetry is installed"
fi

# Check Node.js (for Supabase CLI)
if ! command -v node &> /dev/null; then
    print_warning "Node.js is not installed (needed for Supabase CLI)"
    echo "Install Node.js from: https://nodejs.org/"
else
    print_success "Node.js is installed"
    
    # Check Supabase CLI
    if ! command -v supabase &> /dev/null; then
        print_warning "Supabase CLI is not installed"
        read -p "Install Supabase CLI now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Installing Supabase CLI..."
            npm install -g @supabase/cli
            if command -v supabase &> /dev/null; then
                print_success "Supabase CLI installed successfully"
            else
                print_warning "Supabase CLI installation may have failed"
            fi
        fi
    else
        print_success "Supabase CLI is installed"
    fi
fi

# Install system packages if needed
if [[ "$OS" == "linux" ]]; then
    print_status "Checking Linux system packages..."
    
    packages_needed=()
    
    if ! command -v tesseract &> /dev/null; then
        packages_needed+=("tesseract-ocr")
    fi
    
    if ! command -v redis-cli &> /dev/null; then
        packages_needed+=("redis-tools")
    fi
    
    if ! command -v curl &> /dev/null; then
        packages_needed+=("curl")
    fi
    
    if ! command -v nc &> /dev/null; then
        packages_needed+=("netcat-openbsd")
    fi
    
    if [ ${#packages_needed[@]} -gt 0 ]; then
        print_warning "Missing packages: ${packages_needed[*]}"
        read -p "Install missing packages with apt? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y "${packages_needed[@]}"
        fi
    else
        print_success "All required system packages are installed"
    fi
    
elif [[ "$OS" == "macos" ]]; then
    print_status "Checking macOS system packages..."
    
    if ! command -v tesseract &> /dev/null; then
        print_warning "Tesseract OCR is not installed"
        if command -v brew &> /dev/null; then
            read -p "Install Tesseract with Homebrew? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                brew install tesseract
            fi
        else
            echo "Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
    else
        print_success "Tesseract OCR is installed"
    fi
    
    if ! command -v redis-cli &> /dev/null; then
        print_warning "Redis CLI is not installed"
        if command -v brew &> /dev/null; then
            read -p "Install Redis with Homebrew? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                brew install redis
            fi
        fi
    else
        print_success "Redis CLI is installed"
    fi
fi

# Make scripts executable
print_status "Making scripts executable..."
chmod +x scripts/*.sh
print_success "Scripts are now executable"

# Setup Python environment
print_status "Setting up Python environment..."
poetry install
print_success "Python dependencies installed"

echo ""
echo "============================================"
print_success "System setup completed!"
echo "============================================"
echo ""
echo "📝 Next steps:"
echo "  1. Copy .env.example to .env: cp .env.example .env"
echo "  2. Edit .env file with your configuration"
echo "  3. Start infrastructure: ./scripts/start-local.sh"
echo "  4. Start application: ./scripts/run-dev.sh"
echo "  5. Check health: ./scripts/health-check.sh"
echo ""
echo "🔧 Development workflow:"
echo "  - Run tests: poetry run pytest"
echo "  - Run linting: poetry run ruff check"
echo "  - Format code: poetry run black ."
echo "  - Type check: poetry run mypy src"
echo ""
echo "🧪 Evaluation commands:"
echo "  - CLI evaluation: poetry run python -m src.evaluation.cli run"
echo "  - API evaluation: curl -X POST http://localhost:8080/v1/evaluation/run"
echo ""