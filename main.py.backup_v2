import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_SIZE = 4
HIDDEN_SIZE = 6
LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 100


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(sentence):
    return sentence.lower().split()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(filename):
    with open(filename, "r", encoding="utf-8") as file:
        sentences = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return sentences


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocabulary(tokenized_sentences):

    vocabulary = []

    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    token_to_id = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    id_to_token = {
        index: token
        for token, index in token_to_id.items()
    }

    return token_to_id, id_to_token


# ============================================================
# ENCODE SENTENCES
# ============================================================

def encode_sentences(tokenized_sentences, token_to_id):

    encoded = []

    for sentence in tokenized_sentences:
        encoded_sentence = [
            token_to_id[token]
            for token in sentence
        ]

        encoded.append(encoded_sentence)

    return encoded


# ============================================================
# CREATE EMBEDDING MATRIX
# ============================================================

def create_embedding_matrix(vocabulary_size, embedding_size):

    return np.random.randn(
        vocabulary_size,
        embedding_size
    )


# ============================================================
# RELU
# ============================================================

def relu(x):
    return np.maximum(0, x)


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    # Subtracting the maximum makes the calculation
    # numerically more stable.
    shifted = x - np.max(x)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


# ============================================================
# FORWARD PASS
# ============================================================

def forward_pass(
    input_vector,
    hidden_weights,
    hidden_biases,
    output_weights,
    output_biases
):

    # Hidden layer
    hidden_raw = (
        input_vector @ hidden_weights
        + hidden_biases
    )

    hidden_output = relu(hidden_raw)

    # Output layer
    logits = (
        hidden_output @ output_weights
        + output_biases
    )

    probabilities = softmax(logits)

    return (
        hidden_raw,
        hidden_output,
        logits,
        probabilities
    )


# ============================================================
# TRAIN ONE EXAMPLE
# ============================================================

def train_one_example(
    input_vector,
    target_token_id,
    hidden_weights,
    hidden_biases,
    output_weights,
    output_biases
):

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    (
        hidden_raw,
        hidden_output,
        logits,
        probabilities
    ) = forward_pass(
        input_vector,
        hidden_weights,
        hidden_biases,
        output_weights,
        output_biases
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    target_probability = probabilities[target_token_id]

    loss = -np.log(target_probability)

    # --------------------------------------------------------
    # OUTPUT GRADIENT
    # --------------------------------------------------------

    output_gradient = probabilities.copy()

    output_gradient[target_token_id] -= 1

    # --------------------------------------------------------
    # OUTPUT WEIGHT GRADIENT
    # --------------------------------------------------------

    output_weights_gradient = np.outer(
        hidden_output,
        output_gradient
    )

    # --------------------------------------------------------
    # OUTPUT BIAS GRADIENT
    # --------------------------------------------------------

    output_bias_gradient = output_gradient

    # --------------------------------------------------------
    # HIDDEN GRADIENT
    # --------------------------------------------------------

    hidden_gradient = (
        output_gradient @ output_weights.T
    )

    # ReLU derivative
    relu_gradient = (
        hidden_raw > 0
    ).astype(float)

    hidden_gradient *= relu_gradient

    # --------------------------------------------------------
    # HIDDEN WEIGHT GRADIENT
    # --------------------------------------------------------

    hidden_weights_gradient = np.outer(
        input_vector,
        hidden_gradient
    )

    # --------------------------------------------------------
    # HIDDEN BIAS GRADIENT
    # --------------------------------------------------------

    hidden_bias_gradient = hidden_gradient

    # --------------------------------------------------------
    # WEIGHT UPDATES
    # --------------------------------------------------------

    hidden_weights -= (
        LEARNING_RATE
        * hidden_weights_gradient
    )

    hidden_biases -= (
        LEARNING_RATE
        * hidden_bias_gradient
    )

    output_weights -= (
        LEARNING_RATE
        * output_weights_gradient
    )

    output_biases -= (
        LEARNING_RATE
        * output_bias_gradient
    )

    return (
        loss,
        probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(42)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    sentences = load_training_data("data.txt")

    print(
        f"Loaded {len(sentences)} training sentences."
    )

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    tokenized_sentences = [
        tokenize(sentence)
        for sentence in sentences
    ]

    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    token_to_id, id_to_token = build_vocabulary(
        tokenized_sentences
    )

    vocabulary_size = len(token_to_id)

    print(
        f"Vocabulary size: {vocabulary_size}"
    )

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    encoded_sentences = encode_sentences(
        tokenized_sentences,
        token_to_id
    )

    # --------------------------------------------------------
    # EMBEDDING MATRIX
    # --------------------------------------------------------

    embedding_matrix = create_embedding_matrix(
        vocabulary_size,
        EMBEDDING_SIZE
    )

    # --------------------------------------------------------
    # NEURAL NETWORK
    # --------------------------------------------------------

    hidden_weights = np.random.randn(
        EMBEDDING_SIZE,
        HIDDEN_SIZE
    )

    hidden_biases = np.random.randn(
        HIDDEN_SIZE
    )

    output_weights = np.random.randn(
        HIDDEN_SIZE,
        vocabulary_size
    )

    output_biases = np.random.randn(
        vocabulary_size
    )

    # --------------------------------------------------------
    # OUR TRAINING EXAMPLE
    #
    # "france" -> "paris"
    # --------------------------------------------------------

    france_token_id = token_to_id["france"]
    paris_token_id = token_to_id["paris"]

    france_vector = (
        embedding_matrix[france_token_id]
    )

    print()
    print("# TRAINING EXAMPLE")
    print()
    print(
        f"Input:  france "
        f"(token {france_token_id})"
    )

    print(
        f"Target: paris "
        f"(token {paris_token_id})"
    )

    print()
    print("# TRAINING")

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for iteration in range(1, TRAINING_ITERATIONS + 1):

        loss, probabilities = train_one_example(
            france_vector,
            paris_token_id,
            hidden_weights,
            hidden_biases,
            output_weights,
            output_biases
        )

        predicted_token_id = int(
            np.argmax(probabilities)
        )

        predicted_word = (
            id_to_token[predicted_token_id]
        )

        paris_probability = (
            probabilities[paris_token_id]
        )

        # Print every 10 iterations
        if (
            iteration == 1
            or iteration % 10 == 0
        ):

            print(
                f"Iteration {iteration:3d} | "
                f"Loss: {loss:8.4f} | "
                f"Paris probability: "
                f"{paris_probability:.6f} | "
                f"Prediction: {predicted_word}"
            )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("# FINAL RESULT")
    print()

    print(
        f"Paris probability: "
        f"{paris_probability:.6f}"
    )

    print(
        f"Final loss: "
        f"{loss:.6f}"
    )

    print(
        f"Final prediction: "
        f"{predicted_word}"
    )


if __name__ == "__main__":
    main()