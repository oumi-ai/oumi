# Oumi Analyze Web UI

React-based web interface for viewing analysis results.

## Quick Start

### Prerequisites

- Node.js 18+
- npm

### Build & Run

```bash
# Install dependencies and build
npm install
npm run build

# Then use the CLI to view results
oumi analyze view
```

This serves the React app at `http://localhost:8765` and opens your browser.

## Development

### Dev Server

```bash
# Start development server with hot reload
npm run dev
```

The dev server runs at `http://localhost:5173`.

For the React app to fetch data during development, start the Python data server in another terminal:

```bash
python -m oumi.analyze.serve --port 8765
```

### Rebuild After Changes

After making changes, rebuild:

```bash
npm run build
```

The built files are placed in `dist/` and served by the CLI.

## Architecture

- **Vite + React + TypeScript** - Build toolchain and framework
- **shadcn/ui + Tailwind CSS** - UI components and styling
- **React Query** - Data fetching and caching
- **Recharts** - Charts and visualizations

The app reads JSON data from `/data/index.json` and `/data/evals/{id}.json`, which are served by the Python HTTP server from `~/.oumi/analyze/`.
