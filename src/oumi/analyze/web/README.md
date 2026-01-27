# Oumi Analyze Web UI

React-based web interface for viewing analysis results.

## Development

### Prerequisites

- Node.js 18+
- npm or pnpm

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The dev server runs at `http://localhost:5173` with hot reload.

For the React app to fetch data during development, you need to start the Python data server:

```bash
# In another terminal, start the Python server
python -m oumi.analyze.serve --port 8765
```

### Build

```bash
# Build for production
npm run build
```

The built files are placed in `dist/` and will be served by the CLI.

## Usage

After building, use the CLI to view analysis results:

```bash
oumi analyze view
```

This serves the React app at `http://localhost:8765` and opens your browser.

## Architecture

- **Vite + React + TypeScript** - Build toolchain and framework
- **shadcn/ui + Tailwind CSS** - UI components and styling
- **React Query** - Data fetching and caching
- **Recharts** - Charts and visualizations

The app reads JSON data from `/data/index.json` and `/data/evals/{id}.json`, which are served by the Python HTTP server from `~/.oumi/analyze/`.
