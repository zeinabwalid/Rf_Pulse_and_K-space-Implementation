import matplotlib
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
import scipy.integrate as integrate

import numpy as np
import os.path
import shutil
import FFMPEGwriter

colors = {'bg': [1, 1, 1],
          'circle': [0, 0, 0, .03],
          'axis': [.5, .5, .5],
          'text': [.05, .05, .05],
          'spoilText': [.5, 0, 0],
          'RFtext': [0, .5, 0],
          'Gtext': [80 / 256, 80 / 256, 0],
          'comps': [[.3, .5, .2],
                    [.1, .4, .5],
                    [.5, .3, .2],
                    [.5, .4, .1],
                    [.4, .1, .5],
                    [.6, .1, .3]],
          'boards': {'w1': [.5, 0, 0],
                     'Gx': [0, .5, 0],
                     'Gy': [0, .5, 0],
                     'Gz': [0, .5, 0]
                     }
          }

gyro = 42577.0

initial_state = [0, 0, 1]  # initial state of bulk magnetization vector


class Arrow3D(FancyArrowPatch):
    '''
    Matplotlib FancyArrowPatch for 3D rendering.
    '''

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def derivs(M, t, Meq, w, w1, T1=np.inf, T2=np.inf):
    """
    Bloch equations in rotating frame.
    Args:
        w:              Larmor frequency :math:`2\\pi\\gamma B_0` [kRad / s].
        w1 (complex):   B1 rotation frequency :math:`2\\pi\\gamma B_1`  [kRad / s].
        T1:             longitudinal relaxation time.
        T2:             transverse relaxation time.
        M:              magnetization vector.
        Meq:            equilibrium magnetization.
        t:              time vector (needed for scipy.integrate.odeint).

    Returns:
        integrand :math:`\\frac{dM}{dt}`
    """

    dMdt = np.zeros_like(M)
    dMdt[0] = -M[0] / T2 + M[1] * w + M[2] * w1.real
    dMdt[1] = -M[0] * w - M[1] / T2 + M[2] * w1.imag
    dMdt[2] = -M[0] * w1.real - M[1] * w1.imag + (Meq - M[2]) / T1
    return dMdt


def apply_RF_pluse(dur, FA):
    """
    Apply RF pluse
    Args:
        dur: Duration
        FA: Flip Angle
    Returns:
        Magnetization vector
    """
    print('Applying {} pluse '.format(FA))
    global initial_state
    t = np.linspace(0, dur, 30)
    B1 = np.array([FA / (dur * 360 * gyro * 1e-6)])
    w1 = 2 * np.pi * gyro * B1 * 1e-6
    w1 = w1 * np.exp(1j * 0)  # 0 phase
    Meq = 1
    M = np.zeros([t.size, 3])
    M[0] = initial_state
    M = integrate.odeint(derivs, M[0], t, args=(Meq, 0, w1))  # Solve Bloch equation
    initial_state = M[-1]
    return M


def recovery(dur):
    """
    Simulates Relaxation of Bulk Magnetization vector
    Args:
        dur: Duration of recovery
    Returns:
        Magnetization vector
    """
    print('Recovery')
    global initial_state
    t = np.linspace(0, dur, 100)
    Meq = 1
    T1 = 700 / 1000
    T2 = 300 / 1000
    M = np.zeros([t.size, 3])
    M[0] = initial_state
    M = integrate.odeint(derivs, M[0], t, args=(Meq, 0, 0 + 0j, T1, T2))  # Solve Bloch equation
    initial_state = M[-1]
    return M


def plot_frame3D(vectors, frame, output):
    """
    Creates a plot of magnetization vectors in a 3D view.

    Args:
        frame: which frame to plot.
        vectors: numpy array of size [3, nFrames].
        output: specification of desired output (dictionary from config).
    Returns:
        plot figure.
    """
    # Create 3D axes
    aspect = .952
    figSize = 5  # figure size in inches
    canvasWidth = figSize
    canvasHeight = figSize * aspect
    fig = plt.figure(figsize=(canvasWidth, canvasHeight), dpi=output['dpi'])
    ax = fig.gca(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                 fc=colors['bg'])
    ax.set_axis_off()
    width = 1.65  # to get tight cropping
    height = width / aspect
    left = (1 - width) / 2
    bottom = (1 - height) / 2

    bottom += -.075

    ax.set_position([left, bottom, width, height])

    if output['drawAxes']:
        # Draw axes circles
        for i in ["x", "y", "z"]:
            circle = Circle((0, 0), 1, fill=True, lw=1, fc=colors['circle'])
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

        # Draw x, y, and z axes
        ax.plot([-1, 1], [0, 0], [0, 0], c=colors['axis'], zorder=-1)  # x-axis
        ax.text(1.08, 0, 0, r'$x^\prime$', horizontalalignment='center', color=colors['text'])
        ax.plot([0, 0], [-1, 1], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
        ax.text(0, 1.12, 0, r'$y^\prime$', horizontalalignment='center', color=colors['text'])
        ax.plot([0, 0], [0, 0], [-1, 1], c=colors['axis'], zorder=-1)  # z-axis
        ax.text(0, 0, 1.05, r'$z$', horizontalalignment='center', color=colors['text'])

    pos = [0, 0, 0]
    # Draw magnetization vectors
    M = vectors[frame]

    ax.add_artist(Arrow3D([pos[0], pos[0] + M[0]],
                          [-pos[1], -pos[1] + M[1]],
                          [-pos[2], -pos[2] + M[2]],
                          mutation_scale=20,
                          arrowstyle='-|>', shrinkA=0, shrinkB=0, lw=2,
                          color=colors['comps'][0], alpha=1,
                          zorder=1))
    return fig


def animate(vectors, output_file,leapFactor=1, gifWriter='ffmpeg'):
    """
       simulate magnetization vectors and write animated gif.

       Args:
           leapFactor: Skip frame factor for faster processing and smaller filesize.
           gifWriter:  external program to write gif. Must be "ffmpeg" or "imagemagick"/"convert".
       """
    print('Animate')
    output = {'type': '3D', 'output_file': output_file, 'tRange': [0, 40], 'dpi': 100, 'freeze': [], 'drawAxes': True}

    matplotlib.use('Agg')

    gifWriter = gifWriter.lower()
    if gifWriter == 'ffmpeg':
        if not shutil.which('ffmpeg'):
            raise Exception('FFMPEG not found')
    elif gifWriter in ['imagemagick', 'convert']:
        if not shutil.which('convert'):
            raise Exception('ImageMagick (convert) not found')
    else:
        raise Exception('Argument gifWriter must be "ffmpeg" or "imagemagick"/"convert"')
    delay = int(100 / 15 * leapFactor)  # Delay between frames in ticks of 1/100 sec

    outdir = './out'

    if output['output_file']:
        if gifWriter == 'ffmpeg':
            ffmpegWriter = FFMPEGwriter.FFMPEGwriter(15)
        else:
            tmpdir = './tmp'
            if os.path.isdir(tmpdir):
                rmTmpDir = input('Temporary folder "{}" already exists. Delete(Y/N)?'.format(tmpdir))
                if rmTmpDir.upper() == 'Y':
                    shutil.rmtree(tmpdir)
                else:
                    raise Exception('No files written.')
            os.makedirs(tmpdir, exist_ok=True)

        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, output['output_file'])

        output['freezeFrames'] = []
        for frame in range(0, vectors.shape[0], leapFactor):
            # Use only every leapFactor frame in animation
            if output['type'] == '3D':
                fig = plot_frame3D(vectors, frame, output)
            plt.draw()

            filesToSave = []
            if frame in output['freezeFrames']:
                filesToSave.append('{}_{}.png'.format('.'.join(outfile.split('.')[:-1]), str(frame).zfill(4)))

            if gifWriter == 'ffmpeg':
                ffmpegWriter.addFrame(fig)
            else:  # use imagemagick: save frames temporarily
                filesToSave.append(os.path.join(tmpdir, '{}.png'.format(str(frame).zfill(4))))

            for output_file in filesToSave:
                print('Saving frame {}/{} as "{}"'.format(frame + 1, vectors.shape[0], output_file))
                plt.savefig(output_file, facecolor=plt.gcf().get_facecolor())

            plt.close()
        if gifWriter == 'ffmpeg':
            ffmpegWriter.write(outfile)
        else:  # use imagemagick
            print('Creating animated gif "{}"'.format(outfile))
            compress = '-layers Optimize'
            os.system(('convert {} -delay {} {}/*png {}'.format(compress, delay, tmpdir, outfile)))
            shutil.rmtree(tmpdir)


def plot_trajectory(file_name,data):
    print('plot tajectory')
    matplotlib.use('TkAgg')
    aspect = .952
    figSize = 5  # figure size in inches
    canvasWidth = figSize
    canvasHeight = figSize * aspect
    fig = plt.figure(figsize=(canvasWidth, canvasHeight))
    ax = fig.gca(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                 fc=colors['bg'])
    fig.legend([plt.Line2D((0, 1), (0, 0), lw=2)], ['Bulk Magnetization Trajectory in Rot. Frame'])
    ax.set_axis_off()
    width = 1.65  # to get tight cropping
    height = width / aspect
    left = (1 - width) / 2
    bottom = (1 - height) / 2

    bottom += -.075

    ax.set_position([left, bottom, width, height])
    ax.plot([-1, 1], [0, 0], [0, 0], c=colors['axis'], zorder=-1)  # x-axis
    ax.text(1.08, 0, 0, r'$x^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [-1, 1], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
    ax.text(0, 1.12, 0, r'$y^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [0, 0], [-1, 1], c=colors['axis'], zorder=-1)  # z-axis
    ax.text(0, 0, 1.05, r'$z$', horizontalalignment='center', color=colors['text'])

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.plot(x, y, z)
    plt.savefig('./out/{}'.format(file_name))


if __name__ == '__main__':
    M = apply_RF_pluse(2, 90)
    animate(M, 'RF90.gif')
    plot_trajectory('RF90 trajectory',M)
    M = recovery(3)
    animate(M, 'recovery90.gif')
    plot_trajectory('recovery90 trajectory',M)
    M = apply_RF_pluse(2, 140)
    animate(M, 'RF140.gif')
    plot_trajectory('RF140 trajectory',M)
    M = recovery(4)
    animate(M, 'recovery140.gif')
    plot_trajectory('recovery140 trajectory', M)
    # outputs in ./out directory
