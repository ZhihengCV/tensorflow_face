import os
rootdir = '/media/teddy/data/OFD_full_DB_labelled'


for parent, dirnames, filenames in os.walk(rootdir, topdown=False):
  for dirname in dirnames:
    if dirname == '╣т╒╒':
        pathdir = os.path.join(parent, dirname)
        new_pathdir = os.path.join(parent, 'light')
        os.rename(pathdir, new_pathdir)
    elif dirname == '╩╙╡у':
        pathdir = os.path.join(parent, dirname)
        new_pathdir = os.path.join(parent, 'pose')
        os.rename(pathdir, new_pathdir)
    print(pathdir + ' --> ' + new_pathdir)