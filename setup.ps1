cd ..

wget http://files.minecraftforge.net/maven/net/minecraftforge/forge/1.12.2-14.23.2.2611/forge-1.12.2-14.23.2.2611-mdk.zip -out ../forge.zip
Expand-Archive ../forge.zip ../
rm ../forge.zip

mkdir libs
wget http://svwh.dl.sourceforge.net/project/nujan/Nujan-1.4.2.jar -out libs/Nujan-1.4.2.jar

cp -r ../gradle .
cp ../gradlew .
cp ../gradlew.bat .
cp ../build.gradle .

.\gradlew setupDecompWorkspace
.\gradlew eclipse

cmd /c mklink /d run\python ..\src\main\python

echo "runClient { args '--username', username }" | Out-File build.gradle -append -encoding utf8

cd src
