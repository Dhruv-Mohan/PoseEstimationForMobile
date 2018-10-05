"""
Class to handle landmarkpoints
template pt  {"y":392,"x":311,"point":0,"state":"visible"}
"""


class l_point(object):

    def __init__(self, **kwargs):
        self.__x__ = kwargs['x']
        self.__y__ = kwargs['y']
        '''
        self.__indx__ = kwargs['point']
        if kwargs['state'] in 'visible':
            self.__vis__ = True
        else:
            self.__vis__ = False
        '''
    def get_pt(self):
        return self.__x__, self.__y__

    def index(self):
        return self.__indx__

    def state(self):
        pass

