
'''

'''

class MyDebuger():
    def __init__(self,debug = False,logOutPath = None):
        self.debug = debug
        self.logOutPath = logOutPath
        self.s = ''
        self.ss = []
        self.cutter = ' \n----------------'
    def login(self, *args):
        if self.debug :
            self.clear_s()
            for val in args:
                self.s += str(val)
            self.s += self.cutter

            self.ss.append(self.s)

            if self.logOutPath!=None:
                with open(self.logOutPath,'a')as f:
                    f.writelines(self.s)

    def logout(self):
        if self.debug :
            for s in self.ss:
                print(s)
            print('---------------ending---------------')
            self.clear_ss()

    def clear_ss(self):
        self.ss=[]

    def clear_s(self):
        self.s = ''
