# Deployment Guide

## Overview
This document describes the deployment workflow of the IMDB sentiment classification web application.
We use GitHub Actions as CI/CD pipeline, Docker for containerization, DVC to manage dataset on Azure Blob Storage.

## Pipeline Workflow
1. Trigger: Push or Pull Request to dev / staging / main branch
2. CI test stage:
    - Checkout source code
    - Setup Python environment
    - Install all project dependencies
3. Deploy stage (only triggered on push):
    - Generate environment configuration file (.env) based on branch
    - Build the Docker image for sentiment inference web app
    - Prepare data: pull dataset via DVC from Azure blob

## Environment Configuration
- main branch (production):
  APP_ENV=production
  Use production SAS token to access Azure blob storage
- dev / staging branch (development):
  APP_ENV=staging
  Use staging SAS token to access Azure blob storage

## Local Deployment Steps
1. Clone repository
2. Run `dvc pull` to fetch dataset from Azure
3. Train ML model and save model artifact
4. Build docker image: `docker build -t sentiment-app .`
5. Start container: `docker run -p 8000:8000 --env-file .env sentiment-app`

## Notes
- Dataset version is controlled by DVC
- Model artifacts will be saved in Azure storage
- Secrets (SAS token) should not be committed to git
