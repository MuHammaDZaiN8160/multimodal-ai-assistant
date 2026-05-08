#!/usr/bin/env bash
# Deploy to Azure Container Apps (one-time setup)
# Prerequisites: Azure CLI installed + logged in (az login)

set -e

RESOURCE_GROUP="rg-ai-financial-assistant"
LOCATION="eastus"
ACR_NAME="aifinanacr$(shuf -i 1000-9999 -n1)"   # unique registry name
APP_NAME="ai-financial-assistant"
ENVIRONMENT="env-ai-assistant"

echo "=== Creating resource group ==="
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "=== Creating Azure Container Registry ==="
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

echo "=== Building & pushing image ==="
az acr build --registry $ACR_NAME --image $APP_NAME:latest .

echo "=== Creating Container Apps environment ==="
az containerapp env create \
  --name $ENVIRONMENT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

echo "=== Deploying app ==="
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image "$ACR_NAME.azurecr.io/$APP_NAME:latest" \
  --registry-server "$ACR_NAME.azurecr.io" \
  --registry-username $(az acr credential show -n $ACR_NAME --query username -o tsv) \
  --registry-password $(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv) \
  --target-port 8501 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1 \
  --memory 2Gi \
  --secrets \
      anthropic-key="$ANTHROPIC_API_KEY" \
      groq-key="$GROQ_API_KEY" \
      langchain-key="$LANGCHAIN_API_KEY" \
  --env-vars \
      ANTHROPIC_API_KEY=secretref:anthropic-key \
      GROQ_API_KEY=secretref:groq-key \
      LANGCHAIN_API_KEY=secretref:langchain-key \
      LANGCHAIN_TRACING_V2=true \
      LANGCHAIN_PROJECT=multimodal-ai-assistant

echo ""
echo "=== Deployment complete ==="
az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" -o tsv | xargs -I{} echo "App URL: https://{}"
