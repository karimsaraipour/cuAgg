# Script Configuration ################################################
CORES=32
#######################################################################

# Save original working directory
owd=$(pwd)

# Download & unpack latest cmake
echo "~~ Downloading & Unpacking Latest CMake ~~"
cd $HOME
wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz
tar -xf cmake-*

# Install latest cmake
mkdir $HOME/.local

echo "~~ Installing CMake Locally ~~"
cd cmake-*
./configure --prefix=$HOME/.local --parallel=$CORES -- -DCMAKE_USE_OPENSSL=OFF
gmake -j $CORES
gmake install

# Delete cmake
echo "~~ Cleaning Up Build ~~"
cd $HOME
rm -rf cmake-*

# Return to previous directory
cd $owd

# Update cmake command
echo "export PATH=$HOME/.local/bin:$HOME" >> $HOME/.bashrc
source $HOME/.bashrc
