package edu.usc.thevillagers.serversideagent.request;

import java.io.IOException;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.EntityAgent;
import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.FMLCommonHandler;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

public class RequestManager {
	
	private EnvironmentManager envManager;
	
	private List<Request> requests = new ArrayList<>();
	private long lastRequestTime = 0;
	
	private ServerSocket serv;
	
	public RequestManager(EnvironmentManager envManager) {
		this.envManager = envManager;
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	public void startRequestServer(int port) throws IOException {
		serv = new ServerSocket(port);
		new Thread(() -> {
			while(!serv.isClosed()) {
				try {
					newSocket(new DataSocket(serv.accept()));
				} catch (Exception e) {
					System.out.println("Request failure "+e);
				}
			}
		}).start();
	}
	
	private void newSocket(DataSocket sok) throws Exception {
		sok.socket.setSoTimeout(5000);
		String envClassName = sok.in.readUTF();
		Class<?> envClass = envManager.findEnvClass(envClassName);
		String envId = sok.in.readBoolean() ? null : sok.in.readUTF();
		synchronized(requests) {
			requests.add(new Request(envClass, envId, sok));
			System.out.println("Received request ["+envClass.getSimpleName()+" "+envId+"]");
			lastRequestTime = System.currentTimeMillis();
		}
	}
	
	public void stopRequestServer() throws IOException {
		serv.close();
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		if(event.phase == Phase.START) {
			synchronized(requests) {
				if(System.currentTimeMillis() - lastRequestTime > 100 && !requests.isEmpty()) {
					System.out.println("Processing "+requests.size()+" requests");
					while(!requests.isEmpty())
						processRequest(requests.remove(0));
				}
			}
		}
    }
	
	private void processRequest(Request req) {
		try {
			Environment env;
			if(req.envId != null) {
				env = envManager.getEnv(req.envId);
				if(env == null)
					throw new Exception("No such env "+req.envId);
				if(env.getClass() != req.envClass)
					throw new Exception("Missmatch event types: "+env.getClass()+" | "+req.envClass);
			} else {
				env = envManager.createEnv(req.envClass);
				WorldServer world = FMLCommonHandler.instance().getMinecraftServerInstance().worlds[0];
				if(!env.tryAllocate(world)) throw new Exception("Cannot allocate "+req.envClass);
				env.init(world);
				envManager.registerEnv(env);
			}
			String name = env.id;
			if(name.length() > 16) name = name.substring(0, 16);
			Agent a = new Agent(env, new EntityAgent(env.world, name), req.sok);
			((EntityAgent) a.entity).spawn(env.getOrigin());
			env.newAgent(a);
		} catch (Exception e) {
			System.out.println("Cannot start agent ("+e+")");
		}
	}
	
	public static class Request {
		
		public final Class<?> envClass;
		public final String envId;
		public final DataSocket sok;
		
		public Request(Class<?> envClass, String envId, DataSocket sok) {
			this.envClass = envClass;
			this.envId = envId;
			this.sok = sok;
		}
	}
}
