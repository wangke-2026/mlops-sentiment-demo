# MLOps sentiment demo project
This is the coursework project for DevOps & MLOps, IMDB sentiment classification task.

## Project structure
- app: inference web application
- ml: model training and experiment code
- tests: unit tests
- data: dataset tracked by DVC
- .github/workflows/ci.yml: GitHub Actions CI/CD pipeline

## CI/CD pipeline
- test job: checkout code, setup python, install dependencies
- deploy job: only runs on push event, generate .env file and build docker image
    - For main branch: APP_ENV=production, use production SAS token
    - For dev/staging branch: APP_ENV=staging, use staging SAS token

## How to run
1. Clone repo
2. Pull dataset by DVC
3. Train model
4. Build docker image
5. Start web service

## Notes
Data is stored on Azure blob storage, managed by DVC for data versioning.
