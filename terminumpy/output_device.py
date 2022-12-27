import os
import sys

class OutputDevice:
    def __init__(self, file=sys.stdout, resolution=None):
        ''' Create an OutputDevice. 
        
            Args:
                file: The file to which the output is printed.
                    Defaults to sys.stdout, i.e., probably the terminal.
                resolution (int, int): A (width, height) tuple specifying
                    the resolution of the rendered output
        '''
        self.file = file
        self.resolution = resolution


    def get_resolution(self):
        ''' Returns the resolution. If no resolution was specified during 
            initialization, the current terminal size is returned.
        '''
        if self.resolution is None:
            # Cursor ends up below last line which pushes the first line up
            # Thus, we print one line less
            # to allow removing all printed content
            w, h = os.get_terminal_size()
            return (w, h - 1)
        else:
            return self.resolution
