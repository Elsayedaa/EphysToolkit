import os
import os.path
import re

# this class enables the user to return the absolute paths for files with a certain extension
# within a directory and all its subsequent subdirectories
class path_explorer:
    def __init__(self):
        self.sub_paths = []
        self.sub_files = []
        pass
    
    # returns all directories in a tree
    def subdir(self, path):
        # make an OS compatible path
        path = self.compatible_path(path)
        
        #add the parent path to the sub_paths list
        if path not in self.sub_paths:
            self.sub_paths.append(path)
        
        # for each subcontent in a user given path 
        # if the subcontent are true directories &
        # have not already been added to the sub_paths list
        # append the subcontent to the sub_paths list
        # and recursively run the function on the subcontent
        for sub in os.listdir(path):
            sub_path = os.path.join(path, sub)
            if (os.path.isdir(sub_path)
                    and sub_path not in self.sub_paths):
                self.sub_paths.append(sub_path)
                self.subdir(sub_path)
                
    # helper function for self.subdir to clear the sub_paths list
    def rsubdir(self, path):
        self.sub_paths = []
        self.subdir(path)
        return self.sub_paths
    
    # returns all files of a desired extension in a tree
    def findext(self, path, ext = None, r = None):
        # make a list of sub directories to seach through
        self.rsubdir(path)
        
        # clear the sub_files list
        self.sub_files = []
        
        # for each subconent in the tree sub directories
        # append all the files to the sub_files list if
        # no filter extension is given
        # otherwise, only append files with the given
        # filter extension
        for sub_path in self.sub_paths:
            
            #check the contents of each directory
            contents = os.listdir(sub_path)
            
            for item in contents:
                item_path = os.path.join(sub_path, item)
                name, extension = os.path.splitext(item)
                
                if (ext == None
                       and item_path not in self.sub_files
                       and os.path.isdir(item_path) == False):
                    self.sub_files.append(item_path)
                elif (extension == ext
                           and r == None
                           and item_path not in self.sub_files):
                        self.sub_files.append(item_path)
                elif (extension == ext
                           and re.search(r, item_path)
                           and item_path not in self.sub_files):
                        self.sub_files.append(item_path)
                        
        return self.sub_files
                                         
    #split an absolute path by the host OS seperator
    def recursive_split(self, path, l = []):
        split = os.path.split(path)
        s1 = split[0]
        s2 = split[1]
        if s2 == '':
            l.append(s1)
            l.reverse()
            return l
        else:
            l.append(s2)
            return self.recursive_split(s1, l = l)

    #join a list of path components by the host OS seperator
    def join_split(self, split_list):
        root = split_list[0]
        rest = split_list[1:]
        path_string = root
        for path_component in rest:
            path_string = os.path.join(path_string, path_component)
        return path_string
    
    #returns a path that is compatible with the host OS
    def compatible_path(self, path):
        return self.join_split(self.recursive_split(path, l = []))
