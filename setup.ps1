cd ..

wget http://files.minecraftforge.net/maven/net/minecraftforge/forge/1.12.2-14.23.2.2611/forge-1.12.2-14.23.2.2611-mdk.zip -out ../forge.zip
Expand-Archive ../forge.zip ../
rm ../forge.zip

cp -r ../gradle .
cp ../gradlew .
cp ../gradlew.bat .
cp ../build.gradle .

.\gradlew setupDecompWorkspace
.\gradlew eclipse

cmd /c mklink /d run\python ..\src\main\python

cd src
