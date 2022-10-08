import os
import fnmatch

if __name__ == '__main__':
    for root, dir, files in os.walk('.'):
        print(dir)
