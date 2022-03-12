import os
import sys
import hashlib

def find_duplicates(folder, wrt_file):
    wf = open(wrt_file, 'w+')
    duplicates = {}
    for dirs, subdirs, files in os.walk(folder):
        for name in files:
            file = os.path.join(dirs, name)
            file_hash = hashlib.sha1(open(file, 'rb').read()).digest()
            dup = duplicates.get(file_hash)
            if dup:
                try:
                    duplicates[file_hash][name].append(file)
                except KeyError:
                    duplicates[file_hash][name] = [file]
            else:
                duplicates[file_hash] = {name: [file]}
        for h in duplicates:
            for file in duplicates[h]:
                wf.write('Все файлы: {}'.format(', '.join(duplicates[h][file])) + '\n')
        wf.write('\n')
        for k in duplicates:
            if len(duplicates[k]) > 1:
                wf.write('Дубликаты: {}'.format(', '.join(list(duplicates[k]))) + '\n')
    wf.close()
if __name__ == "__main__":
    find_duplicates(folder = sys.argv[1], wrt_file = (sys.argv[1]+'/'+sys.argv[2]))

