from datasets import load_dataset

### Load User Reviews
## Test avec All_Beauty
dataset_reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)

# Full output
#print(dataset_reviews["full"][0])

# Review informations
print(dataset_reviews['full'][0].keys)

## Load Item Metadata
# Test avec All_Beauty
dataset_items = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
#print(dataset_items[0])

# Item informations
print(dataset_items[0].keys)

