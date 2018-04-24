package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

@SideOnly(value=Side.CLIENT)
public class WorldRecordReplayerClient extends WorldRecordReplayer {
	
	public ReplayWorldAccessClient world;

	public WorldRecordReplayerClient(File saveFolder) {
		super(saveFolder);
	}
	
	@Override
	protected ReplayWorldAccess createWorldAccess() {
		return world = new ReplayWorldAccessClient(from, to);
	}

	@Override
	public void seek(int tick) throws IOException, InterruptedException, ExecutionException {
		super.seek(tick);
		for(int chunkZ = from.getZ() >> 4; chunkZ <= to.getZ() >> 4; chunkZ++)
			for(int chunkY = from.getY() >> 4; chunkY <= to.getY() >> 4; chunkY++)
				for(int chunkX = from.getX() >> 4; chunkX <= to.getX() >> 4; chunkX++)
					world.chunkBufferManager.requestUpdate(chunkX, chunkY, chunkZ);
	}
}
