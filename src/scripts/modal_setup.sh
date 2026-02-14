#!/bin/bash

set -e

echo "JAIDE v40 Modal Setup"
echo "====================="

if ! command -v modal &> /dev/null; then
    echo "Installing Modal CLI..."
    pip install -q modal
fi

echo "Checking Modal authentication..."
if ! modal token check &> /dev/null; then
    echo "Modal token not found. Please authenticate:"
    modal token new
fi

echo "Creating Modal volume for training data..."
modal volume create jaide-training-data || echo "Volume already exists"

echo "Checking Modal secrets..."
if ! modal secret list | grep -q jaide-secrets; then
    echo "Creating jaide-secrets placeholder..."
    echo "JAIDE_API_KEY=placeholder" | modal secret create jaide-secrets
    echo "WARNING: Update jaide-secrets with actual values via: modal secret edit jaide-secrets"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review Modal secrets: modal secret list"
echo "  2. Start training: modal run modal_train.py"
echo "  3. Or distributed: modal run modal_distributed_train.py --epochs 20"
echo "  4. Run inference: modal run modal_inference.py --prompt 'Your prompt here'"
