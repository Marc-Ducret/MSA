package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.world.World;

/**
 * Replayer that uses the real world instead of a 'fake' world. Used for loading an environment state from a recording.
 */
public class WorldRecordReplayerLoad extends WorldRecordReplayer {

	public final Environment env;
	private List<EntityPlayer> actorEntities = new ArrayList<>();
	
	public WorldRecordReplayerLoad(File saveFolder, Environment env) {
		super(saveFolder);
		this.env = env;
	}
	
	@Override
	protected World createWorld() {
		return env.world;
	}
	
	@Override
	public void reset() {
		super.reset();
		actorEntities.clear();
		env.applyToActiveActors((Actor a) -> {
			actorEntities.add(a.entity);
		});
		offset = env.getOrigin().add(-32, -2, -32).subtract(from);
	}
	
	public void spawnEntity(int id, Entity e) {
		idMapping.put(id, e.getEntityId());
		if(!(e instanceof EntityPlayer)) world.spawnEntity(e);
	}
	
	@Override
	public void killEntity(int id) {
		Entity e = world.getEntityByID(idMapping.remove(id));
		if(e == null) {
			System.out.println("Missing entity "+id);
			return;
		}
		if(!(e instanceof EntityPlayer)) world.removeEntityDangerously(e);
	}
	
	@Override
	public EntityPlayer createReplayEntityPlayer(World world, GameProfile profile) {
		if(actorEntities.isEmpty()) throw new RuntimeException("Not enough actors");
		return actorEntities.remove(0);
	}
}
