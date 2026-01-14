# Advanced Trading Bridge 2026 API

## Overview
FastAPI backend for Roblox trading data. Serves as a 100% Roblox-native solution that computes market statistics directly from official Roblox APIs without third-party dependencies.

## Project Type
**FastAPI Web Application**

## Architecture
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Data Source**: Roblox APIs (Economy, Catalog, Inventory, Users)
- **Deployment**: Replit Autoscale / VM

## Features
- **FMV Engine**: Computes Fair Market Value from resale history.
- **Audit Tool**: Analyzes catalog links for "Projected" status and demand.
- **Inventory Scanner**: Fetches and enriches collectible inventories.
- **Search**: Keyword-based catalog search with real-time stats.

## Endpoints
- `GET /market/item/{id}`: Detailed market stats for an asset.
- `POST /market/item/analyze`: Professional audit of a catalog link.
- `GET /market/player/{username}/inventory`: Collectible inventory viewer.
- `GET /market/item/search`: Catalog keyword search.

## Recent Changes
- Switched from Rust library to Python FastAPI backend (January 2026).
- Removed all Rolimons dependencies for 100% Roblox-native data.
- Standardized error reporting and API structure.
