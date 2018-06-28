package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

import edu.usc.thevillagers.serversideagent.command.CommandConstant;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.FMLCommonHandler;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

public class EnvironmentManager {
	
	private Map<String, Environment> envs = new HashMap<>();
	private ExecutorService executor = Executors.newFixedThreadPool(16);
	private List<Function<Phase, Boolean>> tickListeners = new ArrayList<>();
	
	public EnvironmentManager() {
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		if(FMLCommonHandler.instance().getMinecraftServerInstance().worlds[0].getWorldTime() % (CommandConstant.SKIP_TICK+1) == 0) {
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
		List<Future<String>> futures = new ArrayList<>();
		for(Environment env : envs.values()) {
			futures.add(executor.submit(() -> {
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
					return null;
				} catch(Exception e) {
					if(e instanceof ConcurrentModificationException) e.printStackTrace();
					System.err.println("Env "+env.name+" terminated ("+e+")");
					return env.id;
				}
			}));
		}
		for(int i = 0; i < futures.size(); i++) {
			try {
				String envId = futures.get(i).get();
				if(envId != null) {
					envs.get(envId).terminate();
					envs.remove(envId);
				}
			} catch (InterruptedException | ExecutionException e) {
				throw new RuntimeException(e);
			}
		}
		Iterator<Function<Phase, Boolean>> iter = tickListeners.iterator();
		while(iter.hasNext()) {
			try {
				if(iter.next().apply(phase)) iter.remove();
			} catch (Exception e) {
				iter.remove();
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
	
	public Set<String> getEnvIds() {
		return envs.keySet();
	}
	
	public Class<?> findEnvClass(String envName) throws ClassNotFoundException {
		return Class.forName("edu.usc.thevillagers.serversideagent.env.Environment"+envName);
	}
	
	public Environment createEnv(Class<?> envClass) throws InstantiationException, IllegalAccessException {
		return (Environment) envClass.newInstance();
	}
	
	public void addTickListener(Function<Phase, Boolean> listener) {
		tickListeners.add(listener);
	}
}
