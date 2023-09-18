import matplotlib.pyplot as mp

# Initial Values
initvel = 0
initpos = 10
spring = 2
deltaT = 0.05
friction = 0.2

def c_acc(pos0, spring):
    '''change in acc'''
    return -pos0 * spring

def c_vel(vel0, deltaT, a1):
    '''change in vel'''
    return vel0 + deltaT * a1

def c_pos(pos0, vel1, deltaT):
    '''change in pos'''
    return pos0 + vel1 * deltaT

def release_spring(initpos, initvel, deltaT, spring):
    '''initiate spring release'''
    time = [0]
    speeds = [initvel]
    positions = [initpos]
    for i in range(0, 50):
        while time[i] < 50:
            init_acc = c_acc(initpos, spring)
            accelerations = [init_acc-(speeds[i]*friction)]
            initvel = c_vel(initvel, deltaT, accelerations[i])
            speeds = [initvel]
            speeds.append(initvel)
            initpos = c_pos(initpos, speeds[i], deltaT)
            positions.append(initpos)
            time[i] += deltaT
            time.append(time[i])
        return positions, time

release_spring(initpos, initvel, deltaT, spring)
positions, time = release_spring(initpos, initvel, deltaT, spring)
time[0] = 0
mp.plot(time, positions, 'r')
mp.show()