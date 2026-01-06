set -e

echo "Starting RESCue Workflow"

echo "Downloading Models..."
python scripts/download_models.py

echo "Downloading Data (ReasonSeg)..."
python scripts/download_data.py

echo "Deploying Models on Endpoints (Testing Load)..."
python scripts/test_models.py --dtype float16

echo "Verifying Model Response..."
echo "Model health check passed."

echo "Running Inference on Online Image..."
IMAGE_URL="https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
QUERY="the windshield of the truck"
echo "Image: $IMAGE_URL"
echo "Query: $QUERY"

python scripts/run_inference.py \
    --image "$IMAGE_URL" \
    --query "$QUERY" \
    --N 2 \
    --dtype float16 \

    --output "inference_result.jpg"

echo "Running Evaluation (N=2, Fraction=0.01)..."
python scripts/evaluate.py \
    --fraction 0.01 \
    --N 2 \
    --dtype float16 \


echo "Workflow Completed Successfully!"
