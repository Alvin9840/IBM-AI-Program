WATSONX_API_KEY = "YOUR API KEY"
WATSONX_URL = "YOUR WATSON URL"
WATSONX_PROJECT_ID = "YOUR PROJECT ID"

COORDINATOR_ID = "ibm/granite-3-8b-instruct"
COORDINATOR_PARAMETERS = {
    "decoding_method": "greedy",
    "max_new_tokens": 2000,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2
}

PREDICTIVE_ID = "meta-llama/llama-3-3-70b-instruct"
PREDICTIVE_PARAMETERS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1500,
    "temperature": 0.3,
    "repetition_penalty": 1.2
}

SENTIMENT_ID = "meta-llama/llama-3-3-70b-instruct"
SENTIMENT_PARAMETERS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1500,
    "min_new_tokens": 1,
    "temperature": 0.3,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}