#!/usr/bin/python3

import math
import random
import signal
import sys
import time


from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


#TODO: Error checking on txt boxes
#TODO: Color strings


# Import in the code with the actual implementation
from TSPSolver import *
from TSPClasses import *


class PointLineView( QWidget ):
    def __init__( self, status_bar, data_range ):
        super(QWidget,self).__init__()
        self.setMinimumSize(950,600)

        self.pointList  = {}
        #self.lineList   = {}
        self.edgeList   = {}
        self.labelList   = {}
        self.status_bar = status_bar
        self.data_range = data_range
        self.start_pt = None
        self.end_pt = None

    def displayStatusText(self, text):
        self.status_bar.showMessage(text)

    def clearPoints(self):
        #print('POINTS CLEARED!')
        self.pointList = {}

    #def clearLines(self):
        #self.lineList = {}
    def clearEdges(self):
        #pass # TODO
        self.edgeList = {}
        self.labelList = {}
        #self.update()

    def addPoints( self, point_list, color ):
        if color in self.pointList:
            self.pointList[color].extend( point_list )
        else:
            self.pointList[color] = point_list

    def setStartLoc( self, point ):
        self.start_pt = point
        self.update()

    def setEndLoc( self, point ):
        self.end_pt = point
        self.update()

    #def addLines( self, line_list, color ):
        #if color in self.lineList:
            #self.lineList[color].extend( line_list )
        #else:
            #self.lineList[color] = line_list

    def addEdge( self, startPt, endPt, label, edgeColor, labelColor=None, xoffset=0.0 ):
        if not labelColor:
            labelColor = edgeColor

        assert( type(startPt) == QPointF )
        assert( type(endPt)   == QPointF )
        assert( type(label)   == str )

        #if color in self.edgeList:
            #self.lineList[color].extend( line_list )
        #else:
            #self.lineList[color] = line_list

        #print(self.edgeList)
        edge = QLineF(startPt, endPt)
        if edgeColor in self.edgeList.keys():
            self.edgeList[edgeColor].append( edge )
        else:
            self.edgeList[edgeColor] = [edge]
        #self.update()

        midp = QPointF( (edge.x1()*0.2 + edge.x2()*0.8), 
                        (edge.y1()*0.2 + edge.y2()*0.8) )
        self.addLabel( midp, label, labelColor, xoffset=xoffset )

    def addLabel( self, point, label, labelColor,xoffset=0.0 ):
        if labelColor in self.labelList.keys():
            self.labelList[labelColor].append( (point,label,xoffset) )
        else:
            self.labelList[labelColor] = [(point,label,xoffset)]




    def paintEvent(self, event):
        #print('Paint!!!')
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)

        xr = self.data_range['x']
        yr = self.data_range['y']
        w = self.width()
        h = self.height()
        w2h_desired_ratio = (xr[1]-xr[0])/(yr[1]-yr[0])
        #self.aspect_ratio = WIDTH/HEIGHT
        #print('DESIRED', w2h_desired_ratio )
        #print('CURRENT', w/h )
        if w / h < w2h_desired_ratio:
             #h = w / w2h_desired_ratio
             scale = w / (xr[1]-xr[0])
        else:
             #w = h * w2h_desired_ratio
             scale = h / (yr[1]-yr[0])
        #print('FIXED', w/h )
        #print('FIXED', w/h, 'w={},h={},scale={}'.format(w,h,scale) )

        tform = QTransform()
        tform.translate(self.width()/2.0,self.height()/2.0)
        tform.scale(1.0,-1.0)
        painter.setTransform(tform)

        for color in self.edgeList:
            c = QColor(color[0],color[1],color[2])
            painter.setPen( c )
            for edge in self.edgeList[color]:
                ln = QLineF( scale*edge.x1(), scale*edge.y1(), scale*edge.x2(), scale*edge.y2() )
                painter.drawLine( ln )

        for color in self.edgeList:
            c = QColor(color[0],color[1],color[2])
            painter.setPen( c )
            for edge in self.edgeList[color]:
                #arrow_scale = .015
                arrow_scale = 5.0
                unit_edge = ( edge.x2() - edge.x1(), edge.y2() - edge.y1() )
                unit_edge_mag = math.sqrt( ( edge.x2() - edge.x1())**2 + ( edge.y2() - edge.y1() )**2 )
                unit_edge = (unit_edge[0] / unit_edge_mag, unit_edge[1] / unit_edge_mag )
                unit_edge_perp = (-unit_edge[1], unit_edge[0])

                temp_tform = QTransform()
                temp_tform.translate(self.width()/2.0,self.height()/2.0)
                temp_tform.scale(1.0,-1.0)
                temp_tform.translate(scale*edge.x2(),scale*edge.y2())
                temp_tform.scale(1.0,-1.0)
                painter.setTransform(temp_tform)
                #painter.drawText( RECT, label[1], align )

                tri_pts = []
                tri_pts.append( QPointF(0,0) )
                tri_pts.append( QPointF(-arrow_scale*(2*unit_edge[0] + unit_edge_perp[0]),
                                        arrow_scale*(2*unit_edge[1] + unit_edge_perp[1])) )
                tri_pts.append( QPointF(-arrow_scale*(2*unit_edge[0] - unit_edge_perp[0]),
                                        arrow_scale*(2*unit_edge[1] - unit_edge_perp[1])) )
                tri = QPolygonF( tri_pts )
                b = painter.brush()
                painter.setBrush( c )
                painter.drawPolygon( tri )
                painter.setBrush( b )

        painter.setTransform(tform)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)

        R = 1.0E3
        CITY_SIZE = 2.0 # DIAMETER
        RECT = QRectF(-R,-R,2.0*R,2.0*R)
        align = QTextOption( Qt.Alignment(Qt.AlignHCenter | Qt.AlignVCenter) )
        for color in self.labelList:
            c = QColor(color[0],color[1],color[2])
            painter.setPen( c )
            for label in self.labelList[color]:
                temp_tform = QTransform()
                temp_tform.translate(self.width()/2.0,self.height()/2.0)
                temp_tform.scale(1.0,-1.0)
                pt = label[0]
                xoff = label[2]
                temp_tform.translate(scale*pt.x()+xoff,scale*pt.y())
                temp_tform.scale(1.0,-1.0)
                painter.setTransform(temp_tform)
                painter.drawText( RECT, label[1], align )

        painter.setTransform(tform)
        for color in self.pointList:
            c = QColor(color[0],color[1],color[2])
            painter.setPen( c )
            b = painter.brush()
            painter.setBrush(c)
            for point in self.pointList[color]:
                #pt = QPointF(w*point.x(), h*point.y())
                pt = QPointF(scale*point.x(), scale*point.y())
                painter.drawEllipse( pt, CITY_SIZE, CITY_SIZE)
            painter.setBrush(b)

        '''if self.start_pt:
            painter.setPen( QPen(QColor(0.0,255.0,0.0), 2.0) )
            pt = QPointF( scale*self.start_pt.x() -0.0, \
                          scale*self.start_pt.y() -0.0 )
            painter.drawEllipse( pt, 4.0, 4.0)

        if self.end_pt:
            painter.setPen( QPen(QColor(255.0,0.0,0.0), 2.0) )
            pt = QPointF( scale*self.end_pt.x() -0.0, \
                          scale*self.end_pt.y() -0.0 )
            painter.drawEllipse( pt, 4.0, 4.0)'''



class Proj5GUI( QMainWindow ):

    def __init__( self ):
        super(Proj5GUI,self).__init__()

        self.RED_STYLE   = "background-color: rgb(255, 220, 220)"
        self.PLAIN_STYLE = "background-color: rgb(255, 255, 255)"
        self._MAX_SEED = 1000 

        self._scenario = None
        self.initUI()
        self.solver = TSPSolver( self.view )
        self.genParams = {'size':None,'seed':None,'diff':None}


       
    def newPoints(self):

        
        # TODO - ERROR CHECKING!!!!
        #if self.randBySeed.isChecked():
        seed = int(self.curSeed.text())
        random.seed( seed )
        #else: # do by time
            #random.seed( time.time() )

        ptlist = []
        #unique_xvals = {}
        RANGE = self.data_range
        xr = self.data_range['x']
        yr = self.data_range['y']
        npoints = int(self.size.text())
        while len(ptlist) < npoints:
            x = random.uniform(0.0,1.0)
            y = random.uniform(0.0,1.0)
            #if x**2+y**2 <= max_r**2:
            if True:
                xval = xr[0] + (xr[1]-xr[0])*x
                yval = yr[0] + (yr[1]-yr[0])*y
                #if not xval in unique_xvals:
                ptlist.append( QPointF(xval,yval) )
                #print( 'Appended ({},{})'.format(xval,yval) )
                    #unique_xvals[xval] = 1
        return ptlist

    #def clearClicked(self):
        #print('clearClicked')
        #self.view.clearEdges()
        #self.view.update()
        #self.solveButton.setEnabled(True)

    def generateNetwork(self):
        points = self.newPoints() # uses current rand seed
        diff = self.diffDropDown.currentText()
        rand_seed = int(self.curSeed.text())
        self._scenario = Scenario( city_locations=points, difficulty=diff, rand_seed=rand_seed )

        self.genParams = {'size':self.size.text(),'seed':self.curSeed.text(),'diff':diff}
        self.view.clearEdges()
        self.view.clearPoints()

        self.addCities()


    def addCities( self ):
        #X_OFFSET = 0.0375
        cities = self._scenario.getCities()
        self.view.clearEdges()
        for city in cities:
           #self.view.addLabel( city._x, city._y, city._name )
           self.view.addLabel( QPointF(city._x, city._y), city._name, \
                               labelColor=(128,128,128), xoffset=10.0 )

    def generateClicked(self):
        #print('generateClicked')
        if self._scenario:
            result = QMessageBox.question(self, \
                         'Overwrite Existing Scenario?', \
                         'Existing scenario will be overwritten.  Proceed?',
                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if result == QMessageBox.Yes:
                #self.view.clearPoints()
                #self.view.clearLines()
                self.generateNetwork()
                self.view.addPoints( [QPointF(c._x,c._y) for c in self._scenario.getCities()], (0,0,0) )
                self.view.update()
        else:
            self.generateNetwork()
            self.view.addPoints( [QPointF(c._x,c._y) for c in self._scenario.getCities()], (0,0,0) )
            self.view.update()

        #self.solver.reset()
        self.solveButton.setEnabled(True)
        self.graphReady = True
        self.checkGenInputs()
        #self.checkPathInputs()
        self.numSolutions.setText( '--' )
        self.tourCost.setText( '--' )
        self.solvedIn.setText( '--' )
 


    def displaySolution( self ) :
        self.view.clearEdges()
        self.addCities()

        #if array_path:
            #cost = array_path['cost']
            #for start,end,lbl in array_path['path']:
                #self.view.addEdge(startPt=start, endPt=end, label=lbl, edgeColor=(128,128,255))
            #self.arrayTime.setText('{:.6f}s'.format(array_time))
            #if not heap_path:
                #self.heapTime.setText('')
                #self.speedup.setText('')
        edges = self._solution.enumerateEdges()
        if edges:
            edgeColor  = (128,128,255)
            labelColor = (64,64,255)

            for edge in edges:
                pt1,pt2,label = edge
                self.view.addEdge( QPointF(pt1._x,pt1._y), \
                                   QPointF(pt2._x,pt2._y), \
                                   '{}'.format(label), edgeColor, labelColor )

        self.view.update()


    def randSeedClicked(self):
        new_seed = random.randint(0, self._MAX_SEED-1)
        self.curSeed.setText( '{}'.format(new_seed) )

    def solveClicked(self):
        self.solver.setupWithScenario(self._scenario)

        max_time = float( self.timeLimit.text() )
        # TODO - start on a separate thread
        solve_func = 'self.solver.'+self.ALGORITHMS[self.algDropDown.currentIndex()][1]
        results = eval(solve_func)( start_time=time.clock(), time_allowance=max_time )
        #print( results )
        #print( results )
        if results:
            self.numSolutions.setText( '{}'.format(results['count']) )
            self.tourCost.setText( '{}'.format(results['cost']) )
            self.solvedIn.setText( '{:6.6f} seconds'.format(results['time']) )
            self._solution = results['soln']
            #print(self._solution)
            if self._solution:
                self.displaySolution()
        else:
            print( 'GOT NULL SOLUTION BACK!!' )



    #def _randbytime(self):
        #self.randSeed.setEnabled(False)
    #def _randbyseed(self):
        #self.randSeed.setEnabled(True)
    def checkGenInputs(self):
        seed  = self.curSeed.text()
        #print( src )
        size = self.size.text()
        diff = self.diffDropDown.currentText()
        #diff = self.diffDropDown.currentIndex()

        if self._scenario:
            if self.genParams['seed'] == seed and \
               self.genParams['size'] == size and \
               self.genParams['diff'] == diff:
                self.generateButton.setEnabled(False)
                self.solveButton.setEnabled(True)
            elif (seed == '') or (size == ''):
                self.generateButton.setEnabled(False)
                self.solveButton.setEnabled(True)
            else:
                self.generateButton.setEnabled(True)
                self.solveButton.setEnabled(False)


    def checkInputValue(self, widget, validrange):
        assert( type(widget) == QLineEdit )
        retval = None
        valid  = False
        try:
            sval = widget.text()
            #print('Found: \"{}\"'.format(sval))
            if sval == '':
                valid = True
            else:
                ival = int(sval)
                #print('INT: \"{}\"'.format(ival))
                if validrange:
                    #print('HASRANGE:{}'.format(validrange))
                    if ival >= validrange[0] and ival <= validrange[1]:
                        #print('INRANGE!!!')
                        retval = ival
                        valid = True
        except:
            #print( 'INVALID: \"{}\"!!!'.format(sval) )
            pass

        if not valid:
            widget.setStyleSheet( self.RED_STYLE )
            #self.statusBar.showMessage('Invalid value!!!')
        else:
            widget.setStyleSheet( '' )

        return '' if retval==None else retval
            
        
    #def checkPathInputs(self):
        #print( src )
        #if not self.graphReady:
            #self.computeCost.setEnabled(False)
            #print( self.sourceNode.styleSheet() )
            #self.sourceNode.setStyleSheet( '' )
            #self.sourceNode.setEnabled(False)
            #self.targetNode.setStyleSheet( '' )
            #self.targetNode.setEnabled(False)
#
        #else: # HAS GRAPH!!!
            #self.sourceNode.setEnabled(True)
            #self.targetNode.setEnabled(True)
            #self.computeCost.setEnabled(False)
#
            #valid_inds = [1,int(self.genParams[1])]
            #points = self.graph.getNodes()

            #src  = self.checkInputValue( self.sourceNode, valid_inds )
            #if not src == '':
                #self.view.setStartLoc( points[src-1].loc )
            #else:
                #self.view.setStartLoc( None )
#
            #dest = self.checkInputValue( self.targetNode, valid_inds )
            #if not dest == '':
                #if src == dest:
                    #self.targetNode.setStyleSheet( self.RED_STYLE )
                    #self.view.setEndLoc( None )
                #else:
                    #self.view.setEndLoc( points[dest-1].loc )
            #else:
                #self.view.setEndLoc( None )

            
            #if ((not src == self.lastPath[0]) or (not dest == self.lastPath[1])) and \
                #(not src == '') and (not dest == '') and (not src == dest):
                #self.computeCost.setEnabled(True)

        #self.lastPath = (None,None)
        #self.statusBar.showMessage('')

    ALGORITHMS = [ \
        ('Default','defaultRandomTour'), \
        ('Greedy','greedy'), \
        ('Branch and Bound','branchAndBound'), \
        ('Fancy','fancy') \
    ]

    def initUI( self ):
        self.setWindowTitle('Traveling Salesperson Problem')
        self.setWindowIcon( QIcon('icon312.png') )

        self.statusBar = QStatusBar()
        self.setStatusBar( self.statusBar )

        vbox = QVBoxLayout()
        boxwidget = QWidget()
        boxwidget.setLayout(vbox)
        self.setCentralWidget( boxwidget )


        SCALE = 1.0
        self.data_range     = { 'x':[-1.5*SCALE,1.5*SCALE], \
                                'y':[-SCALE,SCALE] }
        self.view           = PointLineView( self.statusBar, \
                                             self.data_range )
        self.randSeedButton = QPushButton('Randomize Seed')
        self.generateButton = QPushButton('Generate Scenario')
        #self.solveButton    = QPushButton('Solve')
        self.solveButton    = QPushButton('Solve TSP')
        #self.clearButton    = QPushButton('Clear To Points')
        '''
        self.useUnsorted    = QRadioButton('Unsorted Array')
        self.useHeap        = QRadioButton('Min Heap')
        self.useBoth        = QRadioButton('Use Both')
        self.arrayTime      = QLineEdit('')
        self.arrayTime.setFixedWidth(120)
        self.arrayTime.setEnabled(False)
        self.heapTime       = QLineEdit('')
        self.heapTime.setFixedWidth(120)
        self.heapTime.setEnabled(False)
        self.speedup        = QLineEdit('')
        self.speedup.setFixedWidth(200)
        self.speedup.setEnabled(False)
        '''

        #self.randByTime     = QRadioButton('Random Seed')
        #self.randBySeed     = QRadioButton('Seed')
        self.curSeed        = QLineEdit('20')
        self.curSeed.setFixedWidth(100)
        self.size           = QLineEdit('15')
        self.size.setFixedWidth(50)
        self.timeLimit      = QLineEdit('60')
        self.timeLimit.setFixedWidth(50)
        self.numSolutions   = QLineEdit('--')
        self.numSolutions.setFixedWidth(100)
        self.tourCost       = QLineEdit('--')
        self.tourCost.setFixedWidth(100)
        self.solvedIn       = QLineEdit('--')

        self.diffDropDown   = QComboBox(self)
        self.algDropDown    = QComboBox(self)

        h = QHBoxLayout()
        h.addWidget( self.view )
        vbox.addLayout(h)

        h = QHBoxLayout()
        h.addWidget( QLabel('Problem Size: ') )
        h.addWidget( self.size )
        h.addWidget( QLabel('Difficulty: ') )
        h.addWidget( self.diffDropDown )
        h.addWidget( QLabel('Current Seed: ') )
        h.addWidget( self.curSeed )
        h.addWidget( self.randSeedButton )
        h.addWidget( self.generateButton )
        #h.addWidget( self.solveButton )
        h.addStretch(1)
        vbox.addLayout(h)
        #self.solveButton.setEnabled(False)

        #h.addWidget( self.clearButton )
        
        h = QHBoxLayout()
        h.addWidget( QLabel('Algorithm: ') )
        h.addWidget( self.algDropDown )
        h.addWidget( QLabel( 'Time Limit' ) )
        h.addWidget( self.timeLimit )
        h.addWidget( QLabel( 'seconds' ) )
        h.addWidget( self.solveButton )
        #self.totalCost.setEnabled(False)
        h.addStretch(1)
        vbox.addLayout(h)

        h = QHBoxLayout()
        h.addWidget( QLabel( '# Solutions:' ) )
        h.addWidget( self.numSolutions )
        h.addWidget( QLabel( 'Cost of tour:' ) )
        h.addWidget( self.tourCost )
        h.addWidget( QLabel( 'Solved in:' ) )
        h.addWidget( self.solvedIn )
        self.numSolutions.setEnabled(False)
        self.tourCost.setEnabled(False)
        self.solvedIn.setEnabled(False)
        h.addStretch(1)
        vbox.addLayout(h)


        #self.randByTime.clicked.connect(self._randbytime)
        #self.randBySeed.clicked.connect(self._randbyseed)


        #self.randBySeed.setChecked(True)
        #self.generateClicked()

        self.lastPath = (None,None)
        self.solveButton.setEnabled(False)
        #self.sourceNode.textChanged.connect(self.checkPathInputs)
        #self.targetNode.textChanged.connect(self.checkPathInputs)

        self.curSeed.textChanged.connect(self.checkGenInputs)
        self.size.textChanged.connect(self.checkGenInputs)

        self.randSeedButton.clicked.connect(self.randSeedClicked)
        self.generateButton.clicked.connect(self.generateClicked)
        self.solveButton.clicked.connect(self.solveClicked)

        #self.diffDropDown.addItem(' ')
        self.diffDropDown.addItem('Easy')
        self.diffDropDown.addItem('Normal')
        self.diffDropDown.addItem('Hard')
        self.diffDropDown.addItem('Hard (Deterministic)')
        #self.diffDropDown.setCurrentIndex(1)
        self.diffDropDown.activated.connect(self.diffChanged)
        self.diffDropDown.setCurrentIndex(3)
        self.diffChanged(3) # to handle start state

        #self.algDropDown.addItem(' ')
        for alg in self.ALGORITHMS:
            self.algDropDown.addItem( alg[0] )
        #self.algDropDown.setCurrentIndex(1)
        self.algDropDown.activated.connect(self.algChanged)
        self.algDropDown.setCurrentIndex(2)
        self.algChanged(2) # to handle start state

        self.graphReady = False
        #self.checkPathInputs()

        self.show()


    def diffChanged(self, text):
        #print('Difficulty changed: now {}'.format(text))
        self.checkGenInputs()

    def algChanged(self, text):
        #print('Algorithm changed: now {}'.format(text))
        pass



if __name__ == '__main__':
    # This line allows CNTL-C in the terminal to kill the program
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    w = Proj5GUI()
    sys.exit(app.exec())
