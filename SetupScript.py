from git import Repo
import os
from shutil import copy as copyFile

def main():
	print("------------------------")
	print("Setup Halloc")
	print("------------------------")

	if not os.path.isdir("frameworks/halloc/repository"):
		Repo.clone_from("https://github.com/canonizer/halloc.git", "frameworks/halloc/repository")
		print("Halloc Repository cloning done!")
	else:
		print("Halloc Repository already cloned")

	print("Overwrite existing files with corrected files!")
	fixed_files_path = "frameworks/halloc/fixes/"
	for file in os.listdir(fixed_files_path):
		filename = fixed_files_path + os.fsdecode(file)
		print("Copy file " + filename)
		copyFile(filename, "frameworks/halloc/repository/src/")
	
	print("------------------------")
	print("Halloc is ready to use!")
	print("------------------------")

	print("------------------------")
	print("Setup ScatterAlloc")
	print("------------------------")

	if not os.path.isdir("frameworks/scatteralloc/repository"):
		Repo.clone_from("https://github.com/ax3l/scatteralloc.git", "frameworks/scatteralloc/repository")
		print("ScatterAlloc Repository cloning done!")
	else:
		print("ScatterAlloc Repository already cloned")

	print("Overwrite existing files with corrected files!")

	print("------------------------")
	print("ScatterAlloc is ready to use!")
	print("------------------------")



if __name__ == "__main__":
	main()