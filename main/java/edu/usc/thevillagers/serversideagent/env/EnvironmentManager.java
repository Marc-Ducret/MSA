package edu.usc.thevillagers.serversideagent.env;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.FMLCommonHandler;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

public class EnvironmentManager {
	
	private Map<String, Environment> envs = new HashMap<>();
	
	public EnvironmentManager() {
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		if(FMLCommonHandler.instance().getMinecraftServerInstance().worlds[0].getWorldTime() % 5 == 0) {
			tickEnvs(event.phase);
		}
    }
	
	public void clearEnvs() {
		for(Environment env : envs.values()) {
			env.terminate();
		}
		envs.clear();
	}
	
	private void tickEnvs(Phase phase) {
    	Iterator<Environment> iter = envs.values().iterator();
		while(iter.hasNext()) {
			Environment env = iter.next();
			try {
				switch(phase) {
				case START:
					if(env.isAllocated() && env.isEmpty()) {
						if(System.currentTimeMillis() - env.allocationTime > 10000) {
							throw new Exception("Empty");
						}
					}
					env.preTick();
					break;
				case END:
					env.postTick();
					break;
				default:
					break;
				}
			} catch(Exception e) {
				env.terminate();
				iter.remove();
				System.err.println("Env "+env.name+" terminated ("+e+")");
			}
		}
    }
	
	public Environment getEnv(String envId) {
		return envs.get(envId);
	}
	
	public boolean doesEnvExists(String envId) {
		return envs.containsKey(envId);
	}
	
	public void registerEnv(Environment env) {
		registerEnv(env, env.name+"#"+env.hashCode());
	}
	
	public void registerEnv(Environment env, String envId) {
		if(doesEnvExists(envId)) throw new RuntimeException("Env "+envId+" already exists");
		envs.put(envId, env);
		env.id = envId;
	}
	
	public void removeEnv(String envId) {
		if(!doesEnvExists(envId)) throw new RuntimeException("Env "+envId+" doesn't exists");
		envs.remove(envId);
	}
	
	public Class<?> findEnvClass(String envName) throws ClassNotFoundException {
		return Class.forName("edu.usc.thevillagers.serversideagent.env.Environment"+envName);
	}
	
	public Environment createEnv(Class<?> envClass) throws InstantiationException, IllegalAccessException {
		return (Environment) envClass.newInstance();
	}
}
