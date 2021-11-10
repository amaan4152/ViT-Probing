def load_weights(weights_file, *args):
    buffer = []
    for f in weights_file.files:
        if not all(keyword in f for keyword in args):
            continue
        print("~ LOADING ===> " + f)
        buffer.append(weights_file[f])
    return buffer
