import os
import dlib
import cv2
import numpy as np
import pickle

def extract_lip_image(img, sx, sy, ex, ey, s, v, u, frame):
	height, width = img.shape[0], img.shape[1]

	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	lip_width, lip_height = ex - sx, ey - sy

	augment_shift = [[-10, -10], [0, -10], [10, -10], [10, 0], [10, 10], [0, 10], [-10, 10], [-10, 0]]
	for i, t in enumerate(augment_shift):
		lip_sx, lip_ex, lip_sy, lip_ey = int((sx + t[0]) / (lip_width / 32)), int((ex + t[0]) / (lip_width / 32)), int((sy + t[1]) / (lip_height / 16)), int((ey + t[1]) / (lip_height / 16))

		if lip_ey > int(height / (lip_height / 16)):
			tmp = lip_ey - int(height / (lip_height / 16))
			lip_sy, lip_ey = lip_sy - tmp, lip_ey - tmp
		
		if lip_ex > int(width / (lip_width / 32)):
			tmp = lip_ex - int(width / (lip_width / 32))
			lip_sx, lip_ex = lip_sx - tmp, lip_ex - tmp

		grayImg = cv2.resize(grayImg, dsize=(int(width / (lip_width / 32)), int(height / (lip_height / 16))), interpolation=cv2.INTER_LINEAR)
		lip = grayImg[lip_sy:lip_ey, lip_sx:lip_ex]


		if not os.path.isdir("./s{0}/s{0}_v{1}_u{2}/{3}".format(s, v, u, i)): os.mkdir("./s{0}/s{0}_v{1}_u{2}/{3}".format(s, v, u, i))

		with open("./s{0}/s{0}_v{1}_u{2}/{3}/{4}.pickle".format(s, v, u, i, frame), 'wb') as f: pickle.dump(lip, f, pickle.HIGHEST_PROTOCOL)

		#좌우 반전
		if not os.path.isdir("./s{0}/s{0}_v{1}_u{2}/{3}".format(s, v, u, i + 8)): os.mkdir("./s{0}/s{0}_v{1}_u{2}/{3}".format(s, v, u, i + 8))
		lip = cv2.flip(lip, 1)
		with open("./s{0}/s{0}_v{1}_u{2}/{3}/{4}.pickle".format(s, v, u, i + 8, frame), 'wb') as f: pickle.dump(lip, f, pickle.HIGHEST_PROTOCOL)

def get_lip_image(s, v, u):
	predictor_path = "../dlib/shape_predictor_68_face_landmarks.dat" # 랜드마크 파일 경로

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	videopath = "../Ouluvs2/original_data/orig_s{0}/orig/s{0}_v{1}_u{2}.mp4".format(s, v, u)
	if not os.path.isdir("./s{0}".format(s)): os.mkdir("./s{0}".format(s))
	if not os.path.isdir("./s{0}/s{0}_v{1}_u{2}".format(s, v, u)): os.mkdir("./s{0}/s{0}_v{1}_u{2}".format(s, v, u))

	capture = cv2.VideoCapture(videopath)

	frame = 0
	sx, sy, ex, ey = -1, 0, 0, 0
	while True:
		ret, img = capture.read()
		if not ret:break

		height, width = img.shape[0], img.shape[1]
		
		dets = detector(img, 0)
		
		if len(dets) == 1:
			for d in dets:
				shape = predictor(img, d)

				px, py = [], []
				for i in range(0, shape.num_parts):		
					px.append(shape.part(i).x)
					py.append(shape.part(i).y)

				sx, ex = np.min(px), np.max(px)
				sy, ey = np.min(py), np.max(py)

				# 입술 가로 크기 옆으로 여유를 준다 (1.5배)
				sx, ex = int(-int((np.min(px[48:])+np.max(px[48:]))/2)*0.5 + np.min(px[48:])*1.5), int(-int((np.min(px[48:])+np.max(px[48:]))/2)*0.5 + np.max(px[48:])*1.5)

				# 세로크기는 중심으로 부터 가로크기의 절반만큼
				sy, ey = int((np.min(py[48:])+np.max(py[48:]))/2) - int((ex-sx)/4), int((np.min(py[48:])+np.max(py[48:]))/2) + int((ex-sx)/4)

				extract_lip_image(img, sx, sy, ex, ey, s, v, u, frame)
				frame = frame + 1
		else:
			if sx < 0:
				print("[*] s{0}_v{1}_u{2} {3} frame Unable to detect face.".format(s, v, u, frame))
			else:
				print("[*] s{0}_v{1}_u{2} {3} frame miss".format(s, v, u, frame))
				extract_lip_image(img, sx, sy, ex, ey, s, v, u, frame)
				frame = frame + 1		
	capture.release()
	cv2.destroyAllWindows()
	return frame

def make_merge_img(cnt, s, u):

	for k in range(16):

		x = []
		for i in range(cnt):
			with open("./s{0}/s{0}_v{1}_u{2}/{3}/{4}.pickle".format(s, 1, u, k, i), 'rb') as f: y = pickle.load(f)
			x.append(y)
		x = np.asarray(x)

		arr = np.zeros([cnt - 1, 3, 16, 32])
		for i in range(cnt - 1):
			arr[i] = np.asarray([x[i], x[i+1], x[i] - x[i+1]])

		with open("./s{0}/s{0}_v{1}_u{2}/{3}.pickle".format(s, 1, u, k), 'wb') as f: pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	for i in range(1, 54):
		if i == 29: continue # 29번은 입술이 보이지 않기 때문에 제외
		for j in range(31, 61):
			frame = get_lip_image(i, 1, j)
			make_merge_img(frame, i, j)
			print("[.] s{0}_v{1}_u{2} ok.".format(i, 1, j))