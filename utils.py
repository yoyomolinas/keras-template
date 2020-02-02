
# import for DirectoryTree
import os
from os.path import join
import shutil

# TODO Define utilities

class DirectoryTree:
	"""
	A class to ease operations in directories
	"""
	def __init__(self, path = None, parent = None, depth = 0, overwrite = False):
		self.parent = parent
		self.path = path
		self.directories = {}
		self.depth = depth
		self.path = path
		if self.depth == 0:
			self.name = self.path
		else:
			self.name = self.path.split('/')[-1]
		if os.path.isfile(self.path):
			raise OSError('Please specify a directory not a file!')

		if os.path.exists(self.path) and overwrite:
			self.remove(hard = True)

		if not os.path.exists(self.path):
			os.makedirs(self.path)
		else:
			# Iterate through all directories in self.path, and add to self.directories
			for dir in os.listdir(self.path):
				if os.path.isdir(join(self.path, dir)):
					self.add(dir)

	def add(self, *names, overwrite = False):
		if not self.exists():
			raise OSError('This directory tree is no longer valid.')
		for name in names:
			if hasattr(self, name) and overwrite:
				self.directories[name].remove(hard = True)
				# raise OSError('path <%s> already exists in this file structure' % join(self.path, name))

			setattr(self, name, DirectoryTree(path = join(self.path, name), parent = self, depth = self.depth + 1))
			self.directories[name] = getattr(self, name)

	def print_all(self):
		if not self.exists():
			raise OSError('This directory tree is no longer valid.')
		cur_path = self.path.split('/')[-1]
		s = ''
		if self.depth != 0:
			s = '|'
			for i in range(self.depth):
				s += '___'

		print("%s%s"%(s, cur_path))
		for name, d in self.directories.items():
			d.print_all()

	def remove(self, hard = False):
		if not self.exists():
			raise OSError('This directory tree is no longer valid.')
		if hard:
			shutil.rmtree(self.path)
		else:
			os.rmdir(self.path)

		if self.parent is not None:
			delattr(self.parent, self.name)
			del self.parent.directories[self.name]

	def exists(self):
		return os.path.isdir(self.path)

	def list_files(self):
		return os.listdir(self.path)
