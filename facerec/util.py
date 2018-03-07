import time
import cv2

import config


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % method.__name__, args, kw, te-ts)
        return result

    return timed


def draw_face(dist, class_id, frame, face):
    x, y, w, h = face
    color = (0, 0, 255)
    name = "Unknown"
    max_dist = 1000000

    print('Distance : ', dist)

    if config.DEBUG:
        print('Distance : ', dist)
        cv2.imwrite('/home/arthur/latest.png', frame)

    if class_id == 69 and dist < config.THRESHOLD:
        color = (0, 255, 0)
        confidence = 100 - (dist / max_dist)
        name = "Bryan Kneis - %2.2f" % confidence

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, name, (x, y - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y + h), (x + w, y), color, 2)


def write_fps(start, end, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = float(1 / float(end - start))
    text = 'Running at %2.2f fps' % fps
    cv2.putText(frame, text, (400, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
