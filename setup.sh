cd ..

sys="$( uname -s )"

case "$sys" in
	Linux*)
		wget http://files.minecraftforge.net/maven/net/minecraftforge/forge/1.12.2-14.23.2.2611/forge-1.12.2-14.23.2.2611-mdk.zip -O ../forge.zip

		;;
	Darwin*)
		curl http://files.minecraftforge.net/maven/net/minecraftforge/forge/1.12.2-14.23.2.2611/forge-1.12.2-14.23.2.2611-mdk.zip -o ../forge.zip
		;;
	*)
esac
unzip ../forge.zip -d ../
rm ../forge.zip

mkdir libs
case "$sys" in 
	Linux*)
		wget http://svwh.dl.sourceforge.net/project/nujan/Nujan-1.4.2.jar -O libs/Nujan-1.4.2.jar
		;;
	Darwin*)
		curl http://svwh.dl.sourceforge.net/project/nujan/Nujan-1.4.2.jar -o libs/Nujan-1.4.2.jar
		;;
	*)
esac

cp -r ../gradle .
cp ../gradlew .
cp ../gradlew.bat .
cp ../build.gradle .

./gradlew setupDecompWorkspace
./gradlew eclipse

ln -s ../src/main/python run/python

echo "runClient { args '--username', username }" >> build.gradle
echo "eula=true" > run/eula.txt
cp src/setup/server.properties run/server.properties

cd src
