import os

# Get the path of running file (.py) in Python: __file__
pasta = os.path.dirname(__file__)
print(pasta)

# Get user path
print(os.path.expanduser('~/Documentos'))

# Set the directory path
# os.chdir(pasta)