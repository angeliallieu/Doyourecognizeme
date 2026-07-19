import os

# Paths
NEG_PATH = os.path.join('data', 'negative')

# Move LFW dataset images to the negative directory
for directory in os.listdir('lfw'):
    if os.path.isdir(os.path.join('lfw', directory)):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)