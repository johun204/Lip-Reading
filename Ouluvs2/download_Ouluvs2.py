import os

# input your credentials
username = ""
password = ""

urls = ["http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/readme",
"http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/cropped_mouth_mp4_phrase.zip",
"http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/cropped_mouth_mp4_digit.zip",
"http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/cropped_audio_dat.zip",
"http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/landmark-sentence.zip",
"http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/transcript_digit_phrase",
"http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/transcript_sentence.zip"]


for url in urls:
	os.system("wget --user " + username + " --password " + password + " " + url)

for sid in range(1, 54):
	if sid == 29: continue # Video data of Subject 29 turned out to be unusable since his mouth was not seen most of the time.
	url = "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/orig_s{}.zip".format(sid)
	os.system("wget --user " + username + " --password " + password + " " + url)
