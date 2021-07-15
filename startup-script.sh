sudo apt-get update
sudo apt-get --assume-yes install git
sudo apt-get install python3-distutils
sudo apt-get install python3-apt

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user

cd prelim
git checkout -b mbm origin/mbm
python3 -m pip install -r requirements.txt --user

git config --global user.email "jclifton333@gmail.com"
git config --global user.name "Jesse Clifton"
