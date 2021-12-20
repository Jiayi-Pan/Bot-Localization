import numpy as np

# motion model
A = np.eye(2)
B = np.eye(2)

# sensor model
C = np.eye(2)

# motion noise covariance
R = np.array([[1, 0],
              [0, 1]])

# sensor noise covariance
Q = np.array([[0.04869528, -0.0058636],
              [-0.0058636, 1.01216104]])

# Path generated from pathGenerator.ipynb
# Path = [[-6.5, 4.0, 0.05], [-6.5, 3.75, 0.05], [-6.5, 3.5, 0.05], [-6.5, 3.25, 0.05], [-6.5, 3.0, 0.05], [-6.5, 2.75, 0.05], [-6.5, 2.5, 0.05], [-6.5, 2.25, 0.05], [-6.5, 2.0, 0.05], [-6.5, 1.75, 0.05], [-6.5, 1.5, 0.05], [-6.5, 1.25, 0.05], [-6.5, 1.0, 0.05], [-6.5, 0.75, 0.05], [-6.5, 0.5, 0.05], [-6.5, 0.25, 0.05], [-6.5, 0.0, 0.05], [-6.5, -0.25, 0.05], [-6.5, -0.5, 0.05], [-6.5, -0.75, 0.05], [-6.5, -1.0, 0.05], [-6.5, -1.25, 0.05], [-6.5, -1.5, 0.05], [-6.5, -1.75, 0.05], [-6.5, -2.0, 0.05], [-6.5, -2.25, 0.05], [-6.5, -2.5, 0.05], [-6.5, -2.75, 0.05], [-6.5, -3.0, 0.05], [-6.5, -3.25, 0.05], [-6.5, -3.5, 0.05], [-6.5, -3.75, 0.05], [-6.5, -4.0, 0.05], [-6.5, -4.25, 0.05], [-6.5, -4.5, 0.05], [-6.5, -4.75, 0.05], [-6.5, -5.0, 0.05], [-6.5, -5.25, 0.05], [-6.5, -5.5, 0.05], [-6.5, -5.75, 0.05], [-6.5, -6.0, 0.05], [-6.5, -6.25, 0.05], [-6.5, -6.5, 0.05], [-6.5, -6.75, 0.05], [-6.5, -7, 0.05], [-6.25, -7, 0.05], [-6.0, -7, 0.05], [-5.75, -7, 0.05], [-5.5, -7, 0.05], [-5.25, -7, 0.05], [-5.0, -7, 0.05], [-4.75, -7, 0.05], [-4.5, -7, 0.05], [-4.25, -7, 0.05], [-4.0, -7, 0.05], [-3.75, -7, 0.05], [-3.5, -7, 0.05], [-3.25, -7, 0.05], [-3.0, -7, 0.05], [-2.75, -7, 0.05], [-2.5, -7, 0.05], [-2.25, -7, 0.05], [-2.0, -7, 0.05], [-1.75, -7, 0.05], [-1.5, -7, 0.05], [-1.25, -7, 0.05], [-1.0, -7, 0.05], [-0.75, -7, 0.05], [-0.5, -7, 0.05], [-0.25, -7, 0.05], [0.0, -7, 0.05], [0.25, -7, 0.05], [0.5, -7, 0.05], [0.75, -7, 0.05], [1, -7.0, 0.05], [1, -6.75, 0.05], [1, -6.5, 0.05], [1, -6.25, 0.05], [1, -6.0, 0.05], [1, -5.75, 0.05], [1, -5.5, 0.05], [1, -5.25, 0.05], [1, -5.0, 0.05], [1, -4.75, 0.05], [1, -4.5, 0.05], [1, -4.25, 0.05], [1, -4.0, 0.05], [1, -3.75, 0.05], [1, -3.5, 0.05], [1, -3.25, 0.05], [1, -3.0, 0.05], [1, -2.75, 0.05], [1, -2.5, 0.05], [1, -2.25, 0.05], [1, -2.0, 0.05], [1, -1.75, 0.05], [1, -1.5, 0.05], [1, -1.25, 0.05], [1, -1.0, 0.05], [1, -0.75, 0.05], [1, -0.5, 0.05], [1, -0.25, 0.05], [1, 0.0, 0.05], [1, 0.25, 0.05], [1.0, 0.5, 0.05], [1.25, 0.5, 0.05], [1.5, 0.5, 0.05], [1.75, 0.5, 0.05], [2.0, 0.5, 0.05], [2.25, 0.5, 0.05], [2.5, 0.5, 0.05], [2.75, 0.5, 0.05], [3.0, 0.5, 0.05], [3.25, 0.5, 0.05], [3.5, 0.5, 0.05], [3.75, 0.5, 0.05], [4.0, 0.5, 0.05], [4.25, 0.5, 0.05], [4.5, 0.5, 0.05], [4.75, 0.5, 0.05], [5.0, 0.5, 0.05], [5.25, 0.5, 0.05], [5.5, 0.5, 0.05], [5.75, 0.5, 0.05], [6.0, 0.5, 0.05], [6.25, 0.5, 0.05], [6.5, 0.5, 0.05], [6.5, 0.75, 0.05], [6.5, 1.0, 0.05], [6.5, 1.25, 0.05], [6.5, 1.5, 0.05], [6.5, 1.75, 0.05], [6.5, 2.0, 0.05], [6.5, 2.25, 0.05], [6.5, 2.5, 0.05], [6.5, 2.75, 0.05], [6.5, 3.0, 0.05], [6.5, 3.25, 0.05], [6.5, 3.5, 0.05], [6.5, 3.75, 0.05], [6.5, 4.0, 0.05], [6.5, 4.25, 0.05], [6.5, 4.5, 0.05], [6.75, 4.5, 0.05], [7.0, 4.5, 0.05], [7.25, 4.5, 0.05], [7.5, 4.5, 0.05], [7.75, 4.5, 0.05], [8.0, 4.5, 0.05], [8.25, 4.5, 0.05], [8.5, 4.5, 0.05], [8.75, 4.5, 0.05], [9.0, 4.5, 0.05], [9.25, 4.5, 0.05], [9.5, 4.5, 0.05], [9.75, 4.5, 0.05], [10.0, 4.5, 0.05], [10.25, 4.5, 0.05], [10.5, 4.5, 0.05], [10.75, 4.5, 0.05], [11.0, 4.5, 0.05], [11.25, 4.5, 0.05], [11.5, 4.5, 0.05], [11.75, 4.5, 0.05], [12.0, 4.5, 0.05], [12.25, 4.5, 0.05], [12.5, 4.5, 0.05], [12.75, 4.5, 0.05]]

# Path_Real = [[-6.484887702394432, 4.0, 0.05], [-6.526198994901157, 3.75, 0.05], [-6.515112297605568, 3.5, 0.05], [-6.473801005098843, 3.25, 0.05], [-6.5112005253950445, 3.0, 0.05], [-6.509849632796448, 2.75, 0.05], [-6.4887994746049555, 2.5, 0.05], [-6.490150367203552, 2.25, 0.05], [-6.416077825924288, 2.0, 0.05], [-6.36955139573067, 1.75, 0.05], [-6.583922174075712, 1.5, 0.05], [-6.63044860426933, 1.25, 0.05], [-6.619224527976042, 1.0, 0.05], [-6.534028319829153, 0.75, 0.05], [-6.380775472023958, 0.5, 0.05], [-6.465971680170847, 0.25, 0.05], [-6.564827153892442, 0.0, 0.05], [-6.421626833512278, -0.25, 0.05], [-6.435172846107558, -0.5, 0.05], [-6.578373166487722, -0.75, 0.05], [-6.593520489989011, -1.0, 0.05], [-6.484068703397353, -1.25, 0.05], [-6.406479510010989, -1.5, 0.05], [-6.515931296602647, -1.75, 0.05], [-6.420879870335126, -2.0, 0.05], [-6.504831549468527, -2.25, 0.05], [-6.579120129664874, -2.5, 0.05], [-6.495168450531473, -2.75, 0.05], [-6.42159845045464, -3.0, 0.05], [-6.439346713188323, -3.25, 0.05], [-6.57840154954536, -3.5, 0.05], [-6.560653286811677, -3.75, 0.05], [-6.614906328521902, -4.0, 0.05], [-6.5544421347265684, -4.25, 0.05], [-6.385093671478098, -4.5, 0.05], [-6.4455578652734316, -4.75, 0.05], [-6.517470556746046, -5.0, 0.05], [-6.534442272952602, -5.25, 0.05], [-6.482529443253954, -5.5, 0.05], [-6.465557727047398, -5.75, 0.05], [-6.464919739890871, -6.0, 0.05], [-6.634211860919552, -6.25, 0.05], [-6.535080260109129, -6.5, 0.05], [-6.365788139080448, -6.75, 0.05], [-6.5, -6.87940861221676, 0.05], [-6.25, -7.059551988278893, 0.05], [-6.0, -7.12059138778324, 0.05], [-5.75, -6.940448011721107, 0.05], [-5.5, -6.950092827413984, 0.05], [-5.25, -6.953747016835306, 0.05], [-5.0, -7.049907172586016, 0.05], [-4.75, -7.046252983164694, 0.05], [-4.5, -7.052941071172855, 0.05], [-4.25, -7.010960719227946, 0.05], [-4.0, -6.947058928827145, 0.05], [-3.75, -6.989039280772054, 0.05], [-3.5, -6.999211475247196, 0.05], [-3.25, -6.976099874991013, 0.05], [-3.0, -7.000788524752804, 0.05], [-2.75, -7.023900125008987, 0.05], [-2.5, -6.899397390518784, 0.05], [-2.25, -6.990831964482774, 0.05], [-2.0, -7.100602609481216, 0.05], [-1.75, -7.009168035517226, 0.05], [-1.5, -6.986179047779819, 0.05], [-1.25, -7.0371296422670415, 0.05], [-1.0, -7.013820952220181, 0.05], [-0.75, -6.9628703577329585, 0.05], [-0.5, -6.978103706654828, 0.05], [-0.25, -7.018372143418477, 0.05], [0.0, -7.021896293345172, 0.05], [0.25, -6.981627856581523, 0.05], [0.5, -6.958240196576885, 0.05], [0.75, -6.915615204053043, 0.05], [0.9648896085557165, -7.0, 0.05], [1.0371505770725038, -6.75, 0.05], [1.0351103914442836, -6.5, 0.05], [0.9628494229274961, -6.25, 0.05], [1.047359818238497, -6.0, 0.05], [1.0885059850997842, -5.75, 0.05], [0.9526401817615028, -5.5, 0.05], [0.9114940149002158, -5.25, 0.05], [1.0938527896903292, -5.0, 0.05], [1.0290533961230655, -4.75, 0.05], [0.9061472103096708, -4.5, 0.05], [0.9709466038769345, -4.25, 0.05], [1.1577679879514404, -4.0, 0.05], [0.9769326098810613, -3.75, 0.05], [0.8422320120485596, -3.5, 0.05], [1.0230673901189387, -3.25, 0.05], [1.0269446401737907, -3.0, 0.05], [1.0110749782657156, -2.75, 0.05], [0.9730553598262093, -2.5, 0.05], [0.9889250217342844, -2.25, 0.05], [0.8245827468044269, -2.0, 0.05], [0.9442410287789499, -1.75, 0.05], [1.175417253195573, -1.5, 0.05], [1.0557589712210502, -1.25, 0.05], [1.0491908024352667, -1.0, 0.05], [0.8591850835688603, -0.75, 0.05], [0.9508091975647334, -0.5, 0.05], [1.1408149164311396, -0.25, 0.05], [1.184668406089474, 0.0, 0.05], [1.0405274417589967, 0.25, 0.05], [1.0, 0.3742006347114525, 0.05], [1.25, 0.4868170875453743, 0.05], [1.5, 0.6257993652885475, 0.05], [1.75, 0.5131829124546258, 0.05], [2.0, 0.47269377239190424, 0.05], [2.25, 0.6241596904924108, 0.05], [2.5, 0.5273062276080958, 0.05], [2.75, 0.3758403095075892, 0.05], [3.0, 0.5684916086404472, 0.05], [3.25, 0.69357700659472, 0.05], [3.5, 0.43150839135955277, 0.05], [3.75, 0.30642299340528, 0.05], [4.0, 0.3913701261553433, 0.05], [4.25, 0.4371040060869103, 0.05], [4.5, 0.6086298738446567, 0.05], [4.75, 0.5628959939130896, 0.05], [5.0, 0.3035255094048085, 0.05], [5.25, 0.5629407458939317, 0.05], [5.5, 0.6964744905951915, 0.05], [5.75, 0.43705925410606833, 0.05], [6.0, 0.3808552144973134, 0.05], [6.25, 0.4339240104966021, 0.05], [6.570673742102088, 0.5, 0.05], [6.547215417685859, 0.75, 0.05], [6.429326257897912, 1.0, 0.05], [6.452784582314141, 1.25, 0.05], [6.564943984244881, 1.5, 0.05], [6.444876399602989, 1.75, 0.05], [6.435056015755119, 2.0, 0.05], [6.555123600397011, 2.25, 0.05], [6.453516110644961, 2.5, 0.05], [6.424525321286712, 2.75, 0.05], [6.546483889355039, 3.0, 0.05], [6.575474678713288, 3.25, 0.05], [6.624640522727435, 3.5, 0.05], [6.451397643432368, 3.75, 0.05], [6.375359477272565, 4.0, 0.05], [6.548602356567632, 4.25, 0.05], [6.5, 4.422306505646993, 0.05], [6.75, 4.422548013997704, 0.05], [7.0, 4.577693494353007, 0.05], [7.25, 4.577451986002296, 0.05], [7.5, 4.455966989421992, 0.05], [7.75, 4.598717943139717, 0.05], [8.0, 4.544033010578008, 0.05], [8.25, 4.401282056860283, 0.05], [8.5, 4.637149603541582, 0.05], [8.75, 4.47168358831335, 0.05], [9.0, 4.362850396458418, 0.05], [9.25, 4.52831641168665, 0.05], [9.5, 4.532120776579234, 0.05], [9.75, 4.414380355395929, 0.05], [10.0, 4.467879223420766, 0.05], [10.25, 4.585619644604071, 0.05], [10.5, 4.632199622669327, 0.05], [10.75, 4.531921402619951, 0.05], [11.0, 4.367800377330673, 0.05], [11.25, 4.468078597380049, 0.05], [11.5, 4.523025091063966, 0.05], [11.75, 4.404683411038979, 0.05], [12.0, 4.476974908936034, 0.05], [12.25, 4.595316588961021, 0.05], [12.5, 4.553160432228537, 0.05], [12.75, 4.697398566999519, 0.05]]

# Path_Real = [[-6.471608312079905, 4.0, 0.05], [-7.0163166842222395, 3.75, 0.05], [-6.528391687920095, 3.5, 0.05], [-5.9836833157777605, 3.25, 0.05], [-6.435150869300746, 3.0, 0.05], [-6.165855378753165, 2.75, 0.05], [-6.564849130699254, 2.5, 0.05], [-6.834144621246835, 2.25, 0.05], [-6.422467066621343, 2.0, 0.05], [-6.680828777197117, 1.75, 0.05], [-6.577532933378657, 1.5, 0.05], [-6.319171222802883, 1.25, 0.05], [-6.255664765912851, 1.0, 0.05], [-6.315842866712796, 0.75, 0.05], [-6.744335234087149, 0.5, 0.05], [-6.684157133287204, 0.25, 0.05], [-6.366870453768724, 0.0, 0.05], [-6.4428985435016255, -0.25, 0.05], [-6.633129546231276, -0.5, 0.05], [-6.5571014564983745, -0.75, 0.05], [-6.10793318534376, -1.0, 0.05], [-6.902704149335759, -1.25, 0.05], [-6.89206681465624, -1.5, 0.05], [-6.097295850664241, -1.75, 0.05], [-6.303495325763042, -2.0, 0.05], [-6.625610312581546, -2.25, 0.05], [-6.696504674236958, -2.5, 0.05], [-6.374389687418454, -2.75, 0.05], [-6.5575592867352475, -3.0, 0.05], [-6.48082832544334, -3.25, 0.05], [-6.4424407132647525, -3.5, 0.05], [-6.51917167455666, -3.75, 0.05], [-6.401006287689632, -4.0, 0.05], [-6.494488956738489, -4.25, 0.05], [-6.598993712310368, -4.5, 0.05], [-6.505511043261511, -4.75, 0.05], [-6.5254456173211395, -5.0, 0.05], [-6.807677371951425, -5.25, 0.05], [-6.4745543826788605, -5.5, 0.05], [-6.192322628048575, -5.75, 0.05], [-6.532348282071833, -6.0, 0.05], [-6.532436435932136, -6.25, 0.05], [-6.467651717928167, -6.5, 0.05], [-6.467563564067864, -6.75, 0.05], [-6.5, -7.129445812834699, 0.05], [-6.25, -7.11435965730858, 0.05], [-6.0, -6.870554187165301, 0.05], [-5.75, -6.88564034269142, 0.05], [-5.5, -7.161344793698937, 0.05], [-5.25, -7.081045173842882, 0.05], [-5.0, -6.838655206301063, 0.05], [-4.75, -6.918954826157118, 0.05], [-4.5, -6.725816917244295, 0.05], [-4.25, -7.169928665315843, 0.05], [-4.0, -7.274183082755705, 0.05], [-3.75, -6.830071334684157, 0.05], [-3.5, -7.041179630774961, 0.05], [-3.25, -7.281070090642971, 0.05], [-3.0, -6.958820369225039, 0.05], [-2.75, -6.718929909357029, 0.05], [-2.5, -7.0264860908519156, 0.05], [-2.25, -6.907815580635225, 0.05], [-2.0, -6.9735139091480844, 0.05], [-1.75, -7.092184419364775, 0.05], [-1.5, -6.86513402150223, 0.05], [-1.25, -6.994954030401332, 0.05], [-1.0, -7.13486597849777, 0.05], [-0.75, -7.005045969598668, 0.05], [-0.5, -7.121998797862856, 0.05], [-0.25, -7.33374803128285, 0.05], [0.0, -6.878001202137144, 0.05], [0.25, -6.66625196871715, 0.05], [0.5, -6.977358628862318, 0.05], [0.75, -7.184045385838728, 0.05], [1.095383454779643, -7.0, 0.05], [0.9547705823916475, -6.75, 0.05], [0.904616545220357, -6.5, 0.05], [1.0452294176083525, -6.25, 0.05], [1.1459932201514906, -6.0, 0.05], [0.6277568411951822, -5.75, 0.05], [0.8540067798485094, -5.5, 0.05], [1.3722431588048178, -5.25, 0.05], [1.0810779252093339, -5.0, 0.05], [0.8201905913865094, -4.75, 0.05], [0.9189220747906661, -4.5, 0.05], [1.1798094086134907, -4.25, 0.05], [1.0885638003206577, -4.0, 0.05], [0.934422059021565, -3.75, 0.05], [0.9114361996793424, -3.5, 0.05], [1.065577940978435, -3.25, 0.05], [1.0883746561293912, -3.0, 0.05], [1.0318852675707764, -2.75, 0.05], [0.9116253438706088, -2.5, 0.05], [0.9681147324292235, -2.25, 0.05], [0.9812249848265095, -2.0, 0.05], [1.3423617787531124, -1.75, 0.05], [1.0187750151734905, -1.5, 0.05], [0.6576382212468876, -1.25, 0.05], [1.1787298643771071, -1.0, 0.05], [0.9706345093383055, -0.75, 0.05], [0.8212701356228929, -0.5, 0.05], [1.0293654906616945, -0.25, 0.05], [0.9843570034249367, 0.0, 0.05], [0.7757682853544826, 0.25, 0.05], [1.0, 0.7144723904092225, 0.05], [1.25, 0.29874393217746636, 0.05], [1.5, 0.28552760959077744, 0.05], [1.75, 0.7012560678225337, 0.05], [2.0, 0.5775172020236442, 0.05], [2.25, 0.6237029559777605, 0.05], [2.5, 0.4224827979763558, 0.05], [2.75, 0.3762970440222395, 0.05], [3.0, 0.319769757283023, 0.05], [3.25, 0.49474413878538964, 0.05], [3.5, 0.6802302427169771, 0.05], [3.75, 0.5052558612146104, 0.05], [4.0, 0.430129864254394, 0.05], [4.25, 0.34821735944611576, 0.05], [4.5, 0.5698701357456061, 0.05], [4.75, 0.6517826405538842, 0.05], [5.0, 0.4115974981189909, 0.05], [5.25, 0.36591434106060006, 0.05], [5.5, 0.5884025018810092, 0.05], [5.75, 0.6340856589393999, 0.05], [6.0, 0.6797951493228102, 0.05], [6.25, 0.888969733824897, 0.05], [6.438062126768174, 0.5, 0.05], [6.271668485327186, 0.75, 0.05], [6.561937873231826, 1.0, 0.05], [6.728331514672814, 1.25, 0.05], [6.228667025267197, 1.5, 0.05], [6.560902563220947, 1.75, 0.05], [6.771332974732803, 2.0, 0.05], [6.439097436779053, 2.25, 0.05], [6.479567789352288, 2.5, 0.05], [6.5690778953241695, 2.75, 0.05], [6.520432210647712, 3.0, 0.05], [6.4309221046758305, 3.25, 0.05], [6.367503114028998, 3.5, 0.05], [6.207858863384031, 3.75, 0.05], [6.632496885971002, 4.0, 0.05], [6.792141136615969, 4.25, 0.05], [6.5, 4.6156809179873575, 0.05], [6.75, 4.224010634858539, 0.05], [7.0, 4.3843190820126425, 0.05], [7.25, 4.775989365141461, 0.05], [7.5, 4.206125154787383, 0.05], [7.75, 4.808940273447051, 0.05], [8.0, 4.793874845212617, 0.05], [8.25, 4.191059726552949, 0.05], [8.5, 4.695583199216853, 0.05], [8.75, 4.360042026666633, 0.05], [9.0, 4.304416800783147, 0.05], [9.25, 4.639957973333367, 0.05], [9.5, 4.277662433326488, 0.05], [9.75, 4.080397271941467, 0.05], [10.0, 4.722337566673512, 0.05], [10.25, 4.919602728058533, 0.05], [10.5, 4.658900958190387, 0.05], [10.75, 4.323667761835414, 0.05], [11.0, 4.341099041809613, 0.05], [11.25, 4.676332238164586, 0.05], [11.5, 4.606092715217662, 0.05], [11.75, 4.622654980532862, 0.05], [12.0, 4.393907284782338, 0.05], [12.25, 4.377345019467138, 0.05], [12.5, 4.331800015643053, 0.05], [12.75, 4.936641293869976, 0.05], [13.0, 4.668199984356947, 0.05], [13.25, 4.063358706130024, 0.05], [13.5, 4.07213772853478, 0.05], [13.75, 4.704950749010279, 0.05]]
Path_Real = [[-6.471608312079905, 4.0, 0.05], [-7.0163166842222395, 3.75, 0.05], [-6.528391687920095, 3.5, 0.05], [-5.9836833157777605, 3.25, 0.05], [-6.435150869300746, 3.0, 0.05], [-6.165855378753165, 2.75, 0.05], [-6.564849130699254, 2.5, 0.05], [-6.834144621246835, 2.25, 0.05], [-6.422467066621343, 2.0, 0.05], [-6.680828777197117, 1.75, 0.05], [-6.577532933378657, 1.5, 0.05], [-6.319171222802883, 1.25, 0.05], [-6.255664765912851, 1.0, 0.05], [-6.315842866712796, 0.75, 0.05], [-6.744335234087149, 0.5, 0.05], [-6.684157133287204, 0.25, 0.05], [-6.366870453768724, 0.0, 0.05], [-6.4428985435016255, -0.25, 0.05], [-6.633129546231276, -0.5, 0.05], [-6.5571014564983745, -0.75, 0.05], [-6.10793318534376, -1.0, 0.05], [-6.902704149335759, -1.25, 0.05], [-6.89206681465624, -1.5, 0.05], [-6.097295850664241, -1.75, 0.05], [-6.303495325763042, -2.0, 0.05], [-6.625610312581546, -2.25, 0.05], [-6.696504674236958, -2.5, 0.05], [-6.374389687418454, -2.75, 0.05], [-6.5575592867352475, -3.0, 0.05], [-6.48082832544334, -3.25, 0.05], [-6.4424407132647525, -3.5, 0.05], [-6.51917167455666, -3.75, 0.05], [-6.401006287689632, -4.0, 0.05], [-6.494488956738489, -4.25, 0.05], [-6.598993712310368, -4.5, 0.05], [-6.505511043261511, -4.75, 0.05], [-6.5254456173211395, -5.0, 0.05], [-6.807677371951425, -5.25, 0.05], [-6.4745543826788605, -5.5, 0.05], [-6.192322628048575, -5.75, 0.05], [-6.532348282071833, -6.0, 0.05], [-6.532436435932136, -6.25, 0.05], [-6.467651717928167, -6.5, 0.05], [-6.467563564067864, -6.75, 0.05], [-6.5, -7.129445812834699, 0.05], [-6.25, -7.11435965730858, 0.05], [-6.0, -6.870554187165301, 0.05], [-5.75, -6.88564034269142, 0.05], [-5.5, -7.161344793698937, 0.05], [-5.25, -7.081045173842882, 0.05], [-5.0, -6.838655206301063, 0.05], [-4.75, -6.918954826157118, 0.05], [-4.5, -6.725816917244295, 0.05], [-4.25, -7.169928665315843, 0.05], [-4.0, -7.274183082755705, 0.05], [-3.75, -6.830071334684157, 0.05], [-3.5, -7.041179630774961, 0.05], [-3.25, -7.281070090642971, 0.05], [-3.0, -6.958820369225039, 0.05], [-2.75, -6.718929909357029, 0.05], [-2.5, -7.0264860908519156, 0.05], [-2.25, -6.907815580635225, 0.05], [-2.0, -6.9735139091480844, 0.05], [-1.75, -7.092184419364775, 0.05], [-1.5, -6.86513402150223, 0.05], [-1.25, -6.994954030401332, 0.05], [-1.0, -7.13486597849777, 0.05], [-0.75, -7.005045969598668, 0.05], [-0.5, -7.121998797862856, 0.05], [-0.25, -7.33374803128285, 0.05], [0.0, -6.878001202137144, 0.05], [0.25, -6.66625196871715, 0.05], [0.5, -6.977358628862318, 0.05], [0.75, -7.184045385838728, 0.05], [1.095383454779643, -7.0, 0.05], [0.9547705823916475, -6.75, 0.05], [0.904616545220357, -6.5, 0.05], [1.0452294176083525, -6.25, 0.05], [1.1459932201514906, -6.0, 0.05], [0.6277568411951822, -5.75, 0.05], [0.8540067798485094, -5.5, 0.05], [1.3722431588048178, -5.25, 0.05], [1.0810779252093339, -5.0, 0.05], [0.8201905913865094, -4.75, 0.05], [0.9189220747906661, -4.5, 0.05], [1.1798094086134907, -4.25, 0.05], [1.0885638003206577, -4.0, 0.05], [0.934422059021565, -3.75, 0.05], [0.9114361996793424, -3.5, 0.05], [1.065577940978435, -3.25, 0.05], [1.0883746561293912, -3.0, 0.05], [1.0318852675707764, -2.75, 0.05], [0.9116253438706088, -2.5, 0.05], [0.9681147324292235, -2.25, 0.05], [0.9812249848265095, -2.0, 0.05], [1.3423617787531124, -1.75, 0.05], [1.0187750151734905, -1.5, 0.05], [0.6576382212468876, -1.25, 0.05], [1.1787298643771071, -1.0, 0.05], [0.9706345093383055, -0.75, 0.05], [0.8212701356228929, -0.5, 0.05], [1.0293654906616945, -0.25, 0.05], [0.9843570034249367, 0.0, 0.05], [0.7757682853544826, 0.25, 0.05], [1.0, 0.7144723904092225, 0.05], [1.25, 0.29874393217746636, 0.05], [1.5, 0.28552760959077744, 0.05], [1.75, 0.7012560678225337, 0.05], [2.0, 0.5775172020236442, 0.05], [2.25, 0.6237029559777605, 0.05], [2.5, 0.4224827979763558, 0.05], [2.75, 0.3762970440222395, 0.05], [3.0, 0.319769757283023, 0.05], [3.25, 0.49474413878538964, 0.05], [3.5, 0.6802302427169771, 0.05], [3.75, 0.5052558612146104, 0.05], [4.0, 0.430129864254394, 0.05], [4.25, 0.34821735944611576, 0.05], [4.5, 0.5698701357456061, 0.05], [4.75, 0.6517826405538842, 0.05], [5.0, 0.4115974981189909, 0.05], [5.25, 0.36591434106060006, 0.05], [5.5, 0.5884025018810092, 0.05], [5.75, 0.6340856589393999, 0.05], [6.0, 0.6797951493228102, 0.05], [6.25, 0.888969733824897, 0.05], [6.438062126768174, 0.5, 0.05], [6.271668485327186, 0.75, 0.05], [6.561937873231826, 1.0, 0.05], [6.728331514672814, 1.25, 0.05], [6.228667025267197, 1.5, 0.05], [6.560902563220947, 1.75, 0.05], [6.771332974732803, 2.0, 0.05], [6.439097436779053, 2.25, 0.05], [6.479567789352288, 2.5, 0.05], [6.5690778953241695, 2.75, 0.05], [6.520432210647712, 3.0, 0.05], [6.4309221046758305, 3.25, 0.05], [6.367503114028998, 3.5, 0.05], [6.207858863384031, 3.75, 0.05], [6.632496885971002, 4.0, 0.05], [6.792141136615969, 4.25, 0.05], [6.5, 4.6156809179873575, 0.05], [6.75, 4.224010634858539, 0.05], [7.0, 4.3843190820126425, 0.05], [7.25, 4.775989365141461, 0.05], [7.5, 4.206125154787383, 0.05], [7.75, 4.808940273447051, 0.05], [8.0, 4.793874845212617, 0.05], [8.25, 4.191059726552949, 0.05], [8.5, 4.695583199216853, 0.05], [8.75, 4.360042026666633, 0.05], [9.0, 4.304416800783147, 0.05], [9.25, 4.639957973333367, 0.05], [9.5, 4.277662433326488, 0.05], [9.75, 4.080397271941467, 0.05], [10.0, 4.722337566673512, 0.05], [10.25, 4.919602728058533, 0.05], [10.5, 4.658900958190387, 0.05], [10.75, 4.323667761835414, 0.05], [11.0, 4.341099041809613, 0.05], [11.25, 4.676332238164586, 0.05], [11.5, 4.606092715217662, 0.05], [11.75, 4.622654980532862, 0.05], [12.0, 4.393907284782338, 0.05], [12.25, 4.377345019467138, 0.05], [12.5, 4.331800015643053, 0.05], [12.75, 4.936641293869976, 0.05]]

Path_Action = [[-6.5, 4.0, 0.05], [-6.5, 3.75, 0.05], [-6.5, 3.5, 0.05], [-6.5, 3.25, 0.05], [-6.5, 3.0, 0.05], [-6.5, 2.75, 0.05], [-6.5, 2.5, 0.05], [-6.5, 2.25, 0.05], [-6.5, 2.0, 0.05], [-6.5, 1.75, 0.05], [-6.5, 1.5, 0.05], [-6.5, 1.25, 0.05], [-6.5, 1.0, 0.05], [-6.5, 0.75, 0.05], [-6.5, 0.5, 0.05], [-6.5, 0.25, 0.05], [-6.5, 0.0, 0.05], [-6.5, -0.25, 0.05], [-6.5, -0.5, 0.05], [-6.5, -0.75, 0.05], [-6.5, -1.0, 0.05], [-6.5, -1.25, 0.05], [-6.5, -1.5, 0.05], [-6.5, -1.75, 0.05], [-6.5, -2.0, 0.05], [-6.5, -2.25, 0.05], [-6.5, -2.5, 0.05], [-6.5, -2.75, 0.05], [-6.5, -3.0, 0.05], [-6.5, -3.25, 0.05], [-6.5, -3.5, 0.05], [-6.5, -3.75, 0.05], [-6.5, -4.0, 0.05], [-6.5, -4.25, 0.05], [-6.5, -4.5, 0.05], [-6.5, -4.75, 0.05], [-6.5, -5.0, 0.05], [-6.5, -5.25, 0.05], [-6.5, -5.5, 0.05], [-6.5, -5.75, 0.05], [-6.5, -6.0, 0.05], [-6.5, -6.25, 0.05], [-6.5, -6.5, 0.05], [-6.5, -6.75, 0.05], [-6.5, -7, 0.05], [-6.25, -7, 0.05], [-6.0, -7, 0.05], [-5.75, -7, 0.05], [-5.5, -7, 0.05], [-5.25, -7, 0.05], [-5.0, -7, 0.05], [-4.75, -7, 0.05], [-4.5, -7, 0.05], [-4.25, -7, 0.05], [-4.0, -7, 0.05], [-3.75, -7, 0.05], [-3.5, -7, 0.05], [-3.25, -7, 0.05], [-3.0, -7, 0.05], [-2.75, -7, 0.05], [-2.5, -7, 0.05], [-2.25, -7, 0.05], [-2.0, -7, 0.05], [-1.75, -7, 0.05], [-1.5, -7, 0.05], [-1.25, -7, 0.05], [-1.0, -7, 0.05], [-0.75, -7, 0.05], [-0.5, -7, 0.05], [-0.25, -7, 0.05], [0.0, -7, 0.05], [0.25, -7, 0.05], [0.5, -7, 0.05], [0.75, -7, 0.05], [1, -7.0, 0.05], [1, -6.75, 0.05], [1, -6.5, 0.05], [1, -6.25, 0.05], [1, -6.0, 0.05], [1, -5.75, 0.05], [1, -5.5, 0.05], [1, -5.25, 0.05], [1, -5.0, 0.05], [1, -4.75, 0.05], [1, -4.5, 0.05], [1, -4.25, 0.05], [1, -4.0, 0.05], [1, -3.75, 0.05], [1, -3.5, 0.05], [1, -3.25, 0.05], [1, -3.0, 0.05], [1, -2.75, 0.05], [1, -2.5, 0.05], [1, -2.25, 0.05], [1, -2.0, 0.05], [1, -1.75, 0.05], [1, -1.5, 0.05], [1, -1.25, 0.05], [1, -1.0, 0.05], [1, -0.75, 0.05], [1, -0.5, 0.05], [1, -0.25, 0.05], [1, 0.0, 0.05], [1, 0.25, 0.05], [1.0, 0.5, 0.05], [1.25, 0.5, 0.05], [1.5, 0.5, 0.05], [1.75, 0.5, 0.05], [2.0, 0.5, 0.05], [2.25, 0.5, 0.05], [2.5, 0.5, 0.05], [2.75, 0.5, 0.05], [3.0, 0.5, 0.05], [3.25, 0.5, 0.05], [3.5, 0.5, 0.05], [3.75, 0.5, 0.05], [4.0, 0.5, 0.05], [4.25, 0.5, 0.05], [4.5, 0.5, 0.05], [4.75, 0.5, 0.05], [5.0, 0.5, 0.05], [5.25, 0.5, 0.05], [5.5, 0.5, 0.05], [5.75, 0.5, 0.05], [6.0, 0.5, 0.05], [6.25, 0.5, 0.05], [6.5, 0.5, 0.05], [6.5, 0.75, 0.05], [6.5, 1.0, 0.05], [6.5, 1.25, 0.05], [6.5, 1.5, 0.05], [6.5, 1.75, 0.05], [6.5, 2.0, 0.05], [6.5, 2.25, 0.05], [6.5, 2.5, 0.05], [6.5, 2.75, 0.05], [6.5, 3.0, 0.05], [6.5, 3.25, 0.05], [6.5, 3.5, 0.05], [6.5, 3.75, 0.05], [6.5, 4.0, 0.05], [6.5, 4.25, 0.05], [6.5, 4.5, 0.05], [6.75, 4.5, 0.05], [7.0, 4.5, 0.05], [7.25, 4.5, 0.05], [7.5, 4.5, 0.05], [7.75, 4.5, 0.05], [8.0, 4.5, 0.05], [8.25, 4.5, 0.05], [8.5, 4.5, 0.05], [8.75, 4.5, 0.05], [9.0, 4.5, 0.05], [9.25, 4.5, 0.05], [9.5, 4.5, 0.05], [9.75, 4.5, 0.05], [10.0, 4.5, 0.05], [10.25, 4.5, 0.05], [10.5, 4.5, 0.05], [10.75, 4.5, 0.05], [11.0, 4.5, 0.05], [11.25, 4.5, 0.05], [11.5, 4.5, 0.05], [11.75, 4.5, 0.05], [12.0, 4.5, 0.05], [12.25, 4.5, 0.05], [12.5, 4.5, 0.05], [12.75, 4.5, 0.05]]
