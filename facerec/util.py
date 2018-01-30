import time
import cv2


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' %
              method.__name__, args, kw, te-ts)
        return result

    return timed


def draw_face(dist, class_id, frame, face):
    x, y, w, h = face
    color = (0, 0, 255)
    name = "Unknown"
    print('Distance : ', dist)
    max = 1000000
    if class_id == 69 and dist < 300000:
        color = (0, 255, 0)
        print('Distance : ', dist)
        confidence = 100 - (dist / max)
        name = "Bryan Kneis - %2.2f" % confidence
        # cv2.imwrite('/home/arthur/latest.png', frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, name, (x, y - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y + h), (x + w, y), color, 2)


def write_fps(start, end, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = float(1 / float(end - start))
    text = 'Running at %2.2f fps' % fps
    cv2.putText(frame, text, (400, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
