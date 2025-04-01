# check if command line argument is empty or not present
if [ "$1" == "" ] || [ $# -gt 1 ]; then
        echo "Usage: run.sh <basename>"
        exit 0;
fi

# strip extension
F="${1%.*}"

# Photosphere Br from WSO
python3 br-from-wso.py 1 ${F}.fits &&

# Run PFSS
python3 fits-to-pot3d.py $F.fits $F.h5 &&
sed "s~FILENAME~$F~g" pot3d.pfss.dat > pot3d.dat &&
../POT3D/bin/pot3d &&
python3 pot3d-to-fits.py ${F}_br_pfss.h5 ${F}_br_pfss.fits &&

# PFCS run
python3 fits-to-pot3d.py ${F}_br_pfss.fits ${F}_br_pfss_2d.h5 &&
sed "s~FILENAME~$F~g" pot3d.pfcs.dat > pot3d.dat &&
../POT3D/bin/pot3d &&
python3 pot3d-to-fits.py ${F}_br_pfcs.h5 ${F}_br_pfcs.fits &&

python3 tracing.py $F &&

python3 wsa2bc_simple.py ${F}_br_pfcs.fits ${F}_br_outer_pfcs_polarity.fits ${F}_v_wsa.fits ${F}.bnd.nc
