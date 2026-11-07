"""Computer Vision HW1 - Ali Owlia."""
from pathlib import Path
import json
import csv
import subprocess
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'inputs'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 42

def u8(image):
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)

def read_image(name, gray=False):
    image = cv2.imread(str(DATA / name), 0 if gray else 1)
    if image is None:
        raise FileNotFoundError(DATA / name)
    return image

def info(image):
    return dict(shape=list(image.shape), dtype=str(image.dtype), minimum=int(image.min()), maximum=int(image.max()), channels=1 if image.ndim == 2 else image.shape[2])

def grid(name, images, titles, cols=3, hist=False):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.4*rows), squeeze=False)
    for ax, im, title in zip(axes.flat, images, titles):
        if hist:
            ax.hist(im.ravel(), bins=256, range=(0,256), color='#366789')
            ax.set_xlim(0,255)
            ax.set_xlabel('Intensity'); ax.set_ylabel('Pixel count')
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
        ax.set_title(title, fontsize=11)
    for ax in list(axes.flat)[len(images):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / (name+'.png'), dpi=160)
    plt.close(fig)

def salt_pepper(image, density, rng):
    # Choose distinct spatial pixels, shared by all color channels.
    result = image.copy()
    count = int(round(image.shape[0]*image.shape[1]*density))
    places = rng.choice(image.shape[0]*image.shape[1], count, replace=False)
    flat = result.reshape((-1,) if image.ndim == 2 else (-1,image.shape[2]))
    flat[places[:count//2]] = 255
    flat[places[count//2:]] = 0
    return result

def spatial_filter(image, kernel):
    # Correlation convention matches OpenCV. Keep signed float results.
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or any(k % 2 == 0 for k in kernel.shape):
        raise ValueError('Kernel dimensions must be odd.')
    h,w = image.shape[:2]
    ph,pw = kernel.shape[0]//2, kernel.shape[1]//2
    padding = ((ph,ph),(pw,pw)) + (((0,0),) if image.ndim == 3 else ())
    padded = np.pad(image.astype(float), padding, mode='edge')
    result = np.zeros_like(image, dtype=float)
    # Each shifted slice contributes one kernel coefficient.
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            result += kernel[y,x]*padded[y:y+h,x:x+w]
    return result

def median_filter(image, size=3):
    if size < 1 or size % 2 == 0:
        raise ValueError('Median size must be positive and odd.')
    p = size//2
    padding = ((p,p),(p,p)) + (((0,0),) if image.ndim == 3 else ())
    padded = np.pad(image,padding,mode='edge')
    h,w = image.shape[:2]
    # Compute the order statistic directly; no library median filter.
    neighbors = np.stack([padded[y:y+h,x:x+w] for y in range(size) for x in range(size)])
    return np.sort(neighbors,axis=0)[size*size//2]

def gaussian_kernel(size=5, sigma=1.0):
    axis = np.arange(size)-size//2
    x,y = np.meshgrid(axis,axis)
    kernel = np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return kernel/kernel.sum()

def psnr(reference, image):
    mse = np.mean((reference.astype(float)-image.astype(float))**2)
    return float('inf') if mse == 0 else float(10*np.log10(255**2/mse))

def save_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def part1():
    image = read_image('Peppers.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    threshold,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    grid('01_types',[image,gray,binary],['RGB display','Grayscale',f'Binary: Otsu = {threshold:.0f}'])
    grid('02_channels',[image[:,:,2],image[:,:,1],image[:,:,0],hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]],['R','G','B','H (0-179)','S','V'])
    lower = np.array([35,65,60]); upper = np.array([85,255,255])
    def mask(im):
        return cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),lower,upper)
    original_mask = mask(image)
    grid('03_green',[image,original_mask,cv2.bitwise_and(image,image,mask=original_mask)],['Original','Green mask','Selected regions'])
    high = u8((gray.astype(float)-gray.mean())*1.5+gray.mean())
    low = u8((gray.astype(float)-gray.mean())*0.5+gray.mean())
    contrasts = [float(np.std(im)) for im in [gray,high,low]]
    assert contrasts[1] > contrasts[0] > contrasts[2]
    grid('04_contrast',[gray,high,low],[f'{n}: RMS = {v:.2f}' for n,v in zip(['Original','High','Low'],contrasts)])
    grid('05_contrast_hist',[gray,high,low],['Original','High contrast','Low contrast'],hist=True)
    dark_hsv = hsv.copy(); dark_hsv[:,:,2] = u8(hsv[:,:,2]*0.3)
    dark = cv2.cvtColor(dark_hsv,cv2.COLOR_HSV2BGR)
    def enhance(equalize):
        channels = cv2.cvtColor(dark,cv2.COLOR_BGR2HSV)
        v = channels[:,:,2]
        channels[:,:,2] = cv2.equalizeHist(v) if equalize else u8((v.astype(float)-v.min())*255/max(1,int(v.max())-int(v.min())))
        return cv2.cvtColor(channels,cv2.COLOR_HSV2BGR)
    equalized,stretched = enhance(True),enhance(False)
    variants = [image,dark,equalized,stretched]
    labels = ['Original','Dark (V x 0.3)','Equalization on V','Stretching on V']
    grid('06_lighting',variants,labels,cols=2)
    grid('07_lighting_hist',[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in variants],labels,cols=2,hist=True)
    masks = [mask(im) for im in variants]
    grid('08_masks',masks,labels,cols=2)
    grid('09_masked',[cv2.bitwise_and(im,im,mask=m) for im,m in zip(variants,masks)],labels,cols=2)
    rows=[]
    for name,m in zip(labels,masks):
        union = np.count_nonzero((m>0)|(original_mask>0))
        rows.append(dict(version=name,pixels=int(np.count_nonzero(m)),iou_to_original=float(np.count_nonzero((m>0)&(original_mask>0))/max(1,union))))
    save_csv('segmentation.csv',rows)
    return dict(image=info(image),threshold=threshold,contrast=contrasts,hsv_lower=lower.tolist(),hsv_upper=upper.tolist(),segmentation=rows)

SX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)
SY = SX.T
LAP = np.array([[0,1,0],[1,-4,1],[0,1,0]],float)

def edges(image, builtin=False):
    if builtin:
        x=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3,borderType=cv2.BORDER_REPLICATE)
        y=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3,borderType=cv2.BORDER_REPLICATE)
        lap=cv2.Laplacian(image,cv2.CV_64F,ksize=1,borderType=cv2.BORDER_REPLICATE)
    else:
        x,y,lap=[spatial_filter(image,k) for k in [SX,SY,LAP]]
    return np.hypot(x,y),lap

def part2():
    raw=read_image('scanned_document.png')
    original=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    rng=np.random.default_rng(SEED)
    noisy={'Salt-pepper':salt_pepper(original,0.05,rng),'Gaussian':u8(original.astype(float)+rng.normal(0,20,original.shape))}
    grid('10_noise',[original,*noisy.values()],['Original','Salt-pepper 5%','Gaussian sigma=20'])
    kernels={'Mean':np.ones((3,3))/9,'Gaussian':gaussian_kernel()}
    rows=[]; differences=[]; median=None
    for noise,im in noisy.items():
        manual={name:u8(spatial_filter(im,k)) for name,k in kernels.items()}
        manual['Median']=median_filter(im)
        builtin={'Mean':u8(cv2.blur(im.astype(float),(3,3),borderType=cv2.BORDER_REPLICATE)),
                 'Gaussian':u8(cv2.GaussianBlur(im.astype(float),(5,5),1.0,borderType=cv2.BORDER_REPLICATE)),
                 'Median':cv2.medianBlur(im,3)}
        rows.append(dict(noise=noise,method='Noisy',manual_psnr=psnr(original,im),opencv_psnr=psnr(original,im)))
        for name in manual:
            rows.append(dict(noise=noise,method=name,manual_psnr=psnr(original,manual[name]),opencv_psnr=psnr(original,builtin[name])))
            diff=np.abs(manual[name].astype(float)-builtin[name].astype(float))
            differences.append(dict(input=noise,filter=name,mae=float(diff.mean()),max_error=float(diff.max())))
            assert diff.max() <= 1
        grid('11_'+noise,[original,im,manual['Mean'],manual['Gaussian'],manual['Median'],builtin['Median']],['Original','Noisy','Manual mean 3x3','Manual Gaussian 5x5','Manual median 3x3','OpenCV median'],cols=3)
        grid('12_'+noise,[*manual.values(),*builtin.values()],['Manual '+x for x in manual]+['OpenCV '+x for x in builtin],cols=3)
        if noise=='Salt-pepper': median=manual['Median']
    for label,im in [('original',original),('noisy',noisy['Salt-pepper']),('restored',median)]:
        manual=edges(im); builtin=edges(im,True)
        for name,a,b in zip(['Sobel','Laplacian'],manual,builtin):
            diff=np.abs(a-b)
            differences.append(dict(input=label,filter=name,mae=float(diff.mean()),max_error=float(diff.max())))
            assert np.allclose(a,b,atol=1e-9)
        # A fixed scale makes the three inputs visually comparable.
        grid('13_edges_'+label,[im,u8(manual[0]/4),u8(np.abs(manual[1])/4),im,u8(builtin[0]/4),u8(np.abs(builtin[1])/4)],['Input','Manual Sobel /4','Manual |Laplacian| /4','Input','OpenCV Sobel /4','OpenCV |Laplacian| /4'])
    save_csv('psnr.csv',rows); save_csv('manual_vs_opencv.csv',differences)
    return dict(raw=info(raw),gray=info(original),psnr=rows,differences=differences)

def video_info(path, count_decode=False):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError('Cannot open '+str(path))
    result=dict(width=int(cap.get(3)),height=int(cap.get(4)),fps=float(cap.get(5)),frames=int(cap.get(7)))
    result['duration']=result['frames']/result['fps']
    if count_decode:
        count=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            assert frame.shape[:2]==(result['height'],result['width'])
            count+=1
        result['decoded_frames']=count
        assert count==result['frames']
    cap.release()
    return result

def part3():
    import imageio_ffmpeg
    source=DATA/'park_video.mp4'; meta=video_info(source)
    rng=np.random.default_rng(SEED+1)
    names=['noisy_video.mp4','filtered_video_1.mp4','filtered_video_2.mp4']
    def writer(name):
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f"{meta['width']}x{meta['height']}",'-r',str(meta['fps']),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/name)]
        return subprocess.Popen(command,stdin=subprocess.PIPE)
    # Use high-quality H.264 for practical size and broad playback support.
    cap=cv2.VideoCapture(str(source)); output=writer(names[0]); count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            output.stdin.write(salt_pepper(frame,0.05,rng).tobytes()); count+=1
    finally:
        cap.release(); output.stdin.close()
    assert output.wait()==0 and count==meta['frames']
    cap=cv2.VideoCapture(str(ROOT/names[0])); writers=[writer(n) for n in names[1:]]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            # Both methods receive the same decoded noisy frame.
            writers[0].stdin.write(cv2.medianBlur(frame,3).tobytes())
            writers[1].stdin.write(cv2.GaussianBlur(frame,(5,5),1,borderType=cv2.BORDER_REPLICATE).tobytes())
    finally:
        cap.release()
        for output in writers: output.stdin.close()
    for output in writers: assert output.wait()==0
    checks={n:video_info(ROOT/n,True) for n in names}
    for check in checks.values():
        assert check['width']==meta['width'] and check['height']==meta['height'] and check['frames']==meta['frames']
        assert abs(check['fps']-meta['fps'])<1e-5
    indices=[0,meta['frames']//2,meta['frames']-1]; rows=[]
    for index in indices:
        frames=[]
        for path in [source,*[ROOT/n for n in names]]:
            cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read(); cap.release()
            assert ok; frames.append(frame)
        grid('14_frame_'+str(index),frames,['Original','Noisy 5%','Median 3x3','Gaussian 5x5'],cols=2)
        rows.append(dict(frame=index,noisy_psnr=psnr(frames[0],frames[1]),median_psnr=psnr(frames[0],frames[2]),gaussian_psnr=psnr(frames[0],frames[3])))
    save_csv('video_sample_psnr.csv',rows)
    return dict(input=meta,outputs=checks,samples=rows,noise_density=0.05,audio='Outputs contain video only')

def test_filters():
    rng=np.random.default_rng(42)
    for shape in [(9,11),(9,11,3)]:
        im=rng.integers(0,256,shape,dtype=np.uint8)
        for size in [(3,3),(5,5),(3,5),(1,1)]:
            kernel=rng.normal(size=size)
            assert np.allclose(spatial_filter(im,kernel),cv2.filter2D(im.astype(float),-1,kernel,borderType=cv2.BORDER_REPLICATE))
        assert np.array_equal(median_filter(im),cv2.medianBlur(im,3))
    im=np.full((100,100,3),127,np.uint8)
    noisy=salt_pepper(im,.05,rng)
    assert np.count_nonzero(np.any(noisy!=im,axis=2))==500
    assert np.count_nonzero(noisy[:,:,0]==0)==250
    print('Filter, boundary, signed-output and noise-density checks passed.')

if __name__=='__main__':
    test_filters()
    result={}
    for name,func in [('part1',part1),('part2',part2),('part3',part3)]:
        print('Running',name,flush=True); result[name]=func()
        (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('All processing and checks completed.',flush=True)
