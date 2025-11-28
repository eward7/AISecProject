"""
Simulated Network Environment for DDoS Attack Testing

This module provides a simulated network environment for testing DDoS attacks
and prevention mechanisms. It includes:
- Virtual network nodes (clients, servers, routers)
- Traffic generation (benign and attack)
- DDoS attack simulation
- Real-time traffic monitoring and logging
"""

import numpy as np
import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
from collections import deque
import random
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficType(Enum):
    """Types of network traffic."""
    BENIGN = "benign"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    HTTP_FLOOD = "http_flood"
    DNS_AMPLIFICATION = "dns_amplification"
    SLOWLORIS = "slowloris"
    ICMP_FLOOD = "icmp_flood"


@dataclass
class NetworkPacket:
    """Represents a network packet in the simulation."""
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str  # TCP, UDP, ICMP
    payload_size: int
    timestamp: float
    flags: Dict[str, bool] = field(default_factory=dict)
    traffic_type: TrafficType = TrafficType.BENIGN
    
    def to_features(self) -> np.ndarray:
        """Convert packet to feature vector for ML model."""
        features = [
            hash(self.source_ip) % 65536,  # Source IP hash
            hash(self.destination_ip) % 65536,  # Dest IP hash
            self.source_port,
            self.destination_port,
            {'TCP': 6, 'UDP': 17, 'ICMP': 1}.get(self.protocol, 0),
            self.payload_size,
            self.flags.get('SYN', 0),
            self.flags.get('ACK', 0),
            self.flags.get('FIN', 0),
            self.flags.get('RST', 0),
            self.flags.get('PSH', 0),
            self.flags.get('URG', 0),
        ]
        return np.array(features, dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol,
            'payload_size': self.payload_size,
            'timestamp': self.timestamp,
            'flags': self.flags,
            'traffic_type': self.traffic_type.value
        }


@dataclass
class FlowStatistics:
    """Statistics for a network flow."""
    flow_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    start_time: float
    packets: List[NetworkPacket] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        if len(self.packets) < 2:
            return 0.0
        return self.packets[-1].timestamp - self.packets[0].timestamp
    
    @property
    def packet_count(self) -> int:
        """Total number of packets in flow."""
        return len(self.packets)
    
    @property
    def total_bytes(self) -> int:
        """Total bytes in flow."""
        return sum(p.payload_size for p in self.packets)
    
    @property
    def packets_per_second(self) -> float:
        """Packets per second."""
        if self.duration == 0:
            return 0.0
        return self.packet_count / self.duration
    
    @property
    def bytes_per_second(self) -> float:
        """Bytes per second."""
        if self.duration == 0:
            return 0.0
        return self.total_bytes / self.duration
    
    def to_features(self) -> np.ndarray:
        """Convert flow statistics to feature vector for ML model."""
        # Calculate inter-arrival times
        if len(self.packets) > 1:
            iats = [self.packets[i+1].timestamp - self.packets[i].timestamp 
                    for i in range(len(self.packets)-1)]
            iat_mean = np.mean(iats)
            iat_std = np.std(iats) if len(iats) > 1 else 0
            iat_max = max(iats)
            iat_min = min(iats)
        else:
            iat_mean = iat_std = iat_max = iat_min = 0
        
        # Packet size statistics
        sizes = [p.payload_size for p in self.packets]
        size_mean = np.mean(sizes) if sizes else 0
        size_std = np.std(sizes) if len(sizes) > 1 else 0
        size_max = max(sizes) if sizes else 0
        size_min = min(sizes) if sizes else 0
        
        # Flag counts
        syn_count = sum(1 for p in self.packets if p.flags.get('SYN', False))
        ack_count = sum(1 for p in self.packets if p.flags.get('ACK', False))
        fin_count = sum(1 for p in self.packets if p.flags.get('FIN', False))
        rst_count = sum(1 for p in self.packets if p.flags.get('RST', False))
        psh_count = sum(1 for p in self.packets if p.flags.get('PSH', False))
        
        features = np.array([
            self.source_port,
            self.destination_port,
            {'TCP': 6, 'UDP': 17, 'ICMP': 1}.get(self.protocol, 0),
            self.duration,
            self.packet_count,
            self.total_bytes,
            self.packets_per_second,
            self.bytes_per_second,
            size_mean,
            size_std,
            size_max,
            size_min,
            iat_mean,
            iat_std,
            iat_max,
            iat_min,
            syn_count,
            ack_count,
            fin_count,
            rst_count,
            psh_count,
        ], dtype=np.float32)
        
        # Pad to 76 features (standard CICDDoS2019 feature count)
        padded = np.zeros(76, dtype=np.float32)
        padded[:len(features)] = features
        
        return padded


class NetworkNode:
    """Represents a node in the simulated network."""
    
    def __init__(self, ip_address: str, node_type: str = "host"):
        self.ip_address = ip_address
        self.node_type = node_type
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()
        self.connections: Dict[str, 'NetworkNode'] = {}
        self.is_active = True
        self.traffic_log: List[NetworkPacket] = []
    
    def send_packet(self, packet: NetworkPacket):
        """Send a packet from this node."""
        self.outgoing_queue.put(packet)
        self.traffic_log.append(packet)
    
    def receive_packet(self, packet: NetworkPacket):
        """Receive a packet at this node."""
        self.incoming_queue.put(packet)
        self.traffic_log.append(packet)
    
    def connect_to(self, node: 'NetworkNode'):
        """Establish connection to another node."""
        self.connections[node.ip_address] = node
        node.connections[self.ip_address] = self


class TrafficGenerator:
    """Generates various types of network traffic."""
    
    def __init__(self, source_ip: str):
        self.source_ip = source_ip
        self._stop_event = threading.Event()
    
    def generate_benign_traffic(
        self,
        destination_ip: str,
        duration: float = 10.0,
        rate: float = 10.0  # packets per second
    ) -> List[NetworkPacket]:
        """
        Generate benign traffic patterns.
        
        Args:
            destination_ip: Target IP address
            duration: Duration in seconds
            rate: Packets per second
            
        Returns:
            List of generated packets
        """
        packets = []
        start_time = time.time()
        interval = 1.0 / rate
        
        current_time = start_time
        while current_time - start_time < duration:
            # Normal web traffic characteristics
            packet = NetworkPacket(
                source_ip=self.source_ip,
                destination_ip=destination_ip,
                source_port=random.randint(49152, 65535),
                destination_port=random.choice([80, 443, 8080]),
                protocol='TCP',
                payload_size=random.randint(64, 1500),
                timestamp=current_time,
                flags={'SYN': False, 'ACK': True, 'FIN': False, 'RST': False, 'PSH': True},
                traffic_type=TrafficType.BENIGN
            )
            packets.append(packet)
            
            # Add some randomness to inter-arrival times
            current_time += interval * random.uniform(0.5, 1.5)
        
        return packets
    
    def generate_syn_flood(
        self,
        destination_ip: str,
        duration: float = 10.0,
        rate: float = 1000.0  # packets per second
    ) -> List[NetworkPacket]:
        """Generate SYN flood attack traffic."""
        packets = []
        start_time = time.time()
        interval = 1.0 / rate
        
        current_time = start_time
        while current_time - start_time < duration:
            # SYN flood: many SYN packets from spoofed IPs
            packet = NetworkPacket(
                source_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
                destination_ip=destination_ip,
                source_port=random.randint(1024, 65535),
                destination_port=random.choice([80, 443, 22, 21]),
                protocol='TCP',
                payload_size=40,  # SYN packets are small
                timestamp=current_time,
                flags={'SYN': True, 'ACK': False, 'FIN': False, 'RST': False, 'PSH': False},
                traffic_type=TrafficType.SYN_FLOOD
            )
            packets.append(packet)
            current_time += interval * random.uniform(0.8, 1.2)
        
        return packets
    
    def generate_udp_flood(
        self,
        destination_ip: str,
        duration: float = 10.0,
        rate: float = 1000.0
    ) -> List[NetworkPacket]:
        """Generate UDP flood attack traffic."""
        packets = []
        start_time = time.time()
        interval = 1.0 / rate
        
        current_time = start_time
        while current_time - start_time < duration:
            packet = NetworkPacket(
                source_ip=f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
                destination_ip=destination_ip,
                source_port=random.randint(1024, 65535),
                destination_port=random.randint(1, 65535),
                protocol='UDP',
                payload_size=random.randint(512, 1472),  # Large UDP packets
                timestamp=current_time,
                flags={},
                traffic_type=TrafficType.UDP_FLOOD
            )
            packets.append(packet)
            current_time += interval * random.uniform(0.8, 1.2)
        
        return packets
    
    def generate_http_flood(
        self,
        destination_ip: str,
        duration: float = 10.0,
        rate: float = 500.0
    ) -> List[NetworkPacket]:
        """Generate HTTP flood attack traffic."""
        packets = []
        start_time = time.time()
        interval = 1.0 / rate
        
        current_time = start_time
        while current_time - start_time < duration:
            # HTTP flood: many legitimate-looking HTTP requests
            packet = NetworkPacket(
                source_ip=self.source_ip,
                destination_ip=destination_ip,
                source_port=random.randint(49152, 65535),
                destination_port=80,
                protocol='TCP',
                payload_size=random.randint(200, 800),  # HTTP request size
                timestamp=current_time,
                flags={'SYN': False, 'ACK': True, 'FIN': False, 'RST': False, 'PSH': True},
                traffic_type=TrafficType.HTTP_FLOOD
            )
            packets.append(packet)
            current_time += interval * random.uniform(0.9, 1.1)
        
        return packets
    
    def generate_dns_amplification(
        self,
        destination_ip: str,
        duration: float = 10.0,
        rate: float = 500.0
    ) -> List[NetworkPacket]:
        """Generate DNS amplification attack traffic."""
        packets = []
        start_time = time.time()
        interval = 1.0 / rate
        
        current_time = start_time
        while current_time - start_time < duration:
            # DNS amplification: large UDP responses
            packet = NetworkPacket(
                source_ip=f"8.8.{random.randint(0,255)}.{random.randint(1,254)}",  # DNS servers
                destination_ip=destination_ip,
                source_port=53,  # DNS port
                destination_port=random.randint(1024, 65535),
                protocol='UDP',
                payload_size=random.randint(2000, 4000),  # Amplified response
                timestamp=current_time,
                flags={},
                traffic_type=TrafficType.DNS_AMPLIFICATION
            )
            packets.append(packet)
            current_time += interval * random.uniform(0.8, 1.2)
        
        return packets


class SimulatedNetwork:
    """
    Simulated network environment for DDoS testing.
    
    This class creates a virtual network with nodes, traffic generation,
    and monitoring capabilities for testing DDoS detection systems.
    """
    
    def __init__(self, network_config: Optional[Dict] = None):
        """
        Initialize the simulated network.
        
        Args:
            network_config: Optional configuration for the network
        """
        self.nodes: Dict[str, NetworkNode] = {}
        self.flows: Dict[str, FlowStatistics] = {}
        self.traffic_log: List[NetworkPacket] = []
        self.detection_callback: Optional[Callable] = None
        self.is_running = False
        self._lock = threading.Lock()
        
        # Default network configuration
        self.config = network_config or {
            'server_ip': '192.168.1.1',
            'client_ips': ['192.168.1.100', '192.168.1.101', '192.168.1.102'],
            'attacker_ips': ['10.0.0.1', '10.0.0.2'],
            'flow_timeout': 120.0  # seconds
        }
        
        self._setup_network()
    
    def _setup_network(self):
        """Set up the network topology."""
        # Create server node
        self.nodes[self.config['server_ip']] = NetworkNode(
            self.config['server_ip'], 
            node_type='server'
        )
        
        # Create client nodes
        for client_ip in self.config['client_ips']:
            self.nodes[client_ip] = NetworkNode(client_ip, node_type='client')
            self.nodes[client_ip].connect_to(self.nodes[self.config['server_ip']])
        
        # Create attacker nodes
        for attacker_ip in self.config['attacker_ips']:
            self.nodes[attacker_ip] = NetworkNode(attacker_ip, node_type='attacker')
        
        logger.info(f"Network initialized with {len(self.nodes)} nodes")
    
    def inject_traffic(self, packets: List[NetworkPacket]):
        """
        Inject traffic into the network.
        
        Args:
            packets: List of packets to inject
        """
        with self._lock:
            for packet in packets:
                self.traffic_log.append(packet)
                
                # Update flow statistics
                flow_id = self._get_flow_id(packet)
                if flow_id not in self.flows:
                    self.flows[flow_id] = FlowStatistics(
                        flow_id=flow_id,
                        source_ip=packet.source_ip,
                        destination_ip=packet.destination_ip,
                        source_port=packet.source_port,
                        destination_port=packet.destination_port,
                        protocol=packet.protocol,
                        start_time=packet.timestamp
                    )
                self.flows[flow_id].packets.append(packet)
                
                # Trigger detection callback if set
                if self.detection_callback:
                    self.detection_callback(packet)
    
    def _get_flow_id(self, packet: NetworkPacket) -> str:
        """Generate unique flow ID for a packet."""
        return f"{packet.source_ip}:{packet.source_port}-{packet.destination_ip}:{packet.destination_port}-{packet.protocol}"
    
    def get_flow_features(self) -> np.ndarray:
        """
        Get feature vectors for all flows.
        
        Returns:
            Array of flow features
        """
        features = []
        for flow in self.flows.values():
            if flow.packet_count > 0:
                features.append(flow.to_features())
        return np.array(features) if features else np.array([]).reshape(0, 76)
    
    def get_flow_labels(self) -> np.ndarray:
        """
        Get labels for all flows (0: benign, 1: attack).
        
        Returns:
            Array of labels
        """
        labels = []
        for flow in self.flows.values():
            if flow.packet_count > 0:
                # Check if any packet in flow is an attack
                is_attack = any(
                    p.traffic_type != TrafficType.BENIGN 
                    for p in flow.packets
                )
                labels.append(1 if is_attack else 0)
        return np.array(labels)
    
    def set_detection_callback(self, callback: Callable):
        """Set callback function for real-time detection."""
        self.detection_callback = callback
    
    def clear(self):
        """Clear all traffic and flow data."""
        with self._lock:
            self.traffic_log.clear()
            self.flows.clear()
    
    def get_statistics(self) -> Dict:
        """Get network statistics."""
        total_packets = len(self.traffic_log)
        benign_packets = sum(1 for p in self.traffic_log if p.traffic_type == TrafficType.BENIGN)
        attack_packets = total_packets - benign_packets
        
        return {
            'total_packets': total_packets,
            'benign_packets': benign_packets,
            'attack_packets': attack_packets,
            'total_flows': len(self.flows),
            'total_bytes': sum(p.payload_size for p in self.traffic_log),
            'attack_types': dict(self._count_attack_types())
        }
    
    def _count_attack_types(self) -> Dict[str, int]:
        """Count packets by attack type."""
        counts = {}
        for packet in self.traffic_log:
            attack_type = packet.traffic_type.value
            counts[attack_type] = counts.get(attack_type, 0) + 1
        return counts
    
    def export_to_csv(self, filepath: str):
        """Export traffic data to CSV format."""
        import pandas as pd
        
        records = [p.to_dict() for p in self.traffic_log]
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(records)} packets to {filepath}")
    
    def export_flows_to_csv(self, filepath: str):
        """Export flow features to CSV format."""
        import pandas as pd
        
        features = self.get_flow_features()
        labels = self.get_flow_labels()
        
        # Create DataFrame with feature names
        columns = [f'feature_{i}' for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=columns)
        df['label'] = labels
        
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} flows to {filepath}")


class AttackScenario:
    """Predefined attack scenarios for testing."""
    
    def __init__(self, network: SimulatedNetwork):
        self.network = network
        self.generator = TrafficGenerator("10.0.0.1")
    
    def run_scenario(
        self,
        scenario_name: str,
        duration: float = 60.0,
        attack_start: float = 20.0,
        attack_duration: float = 20.0
    ) -> Dict:
        """
        Run a predefined attack scenario.
        
        Args:
            scenario_name: Name of the scenario
            duration: Total scenario duration
            attack_start: When to start the attack
            attack_duration: Duration of the attack
            
        Returns:
            Scenario results
        """
        logger.info(f"Starting scenario: {scenario_name}")
        
        target_ip = self.network.config['server_ip']
        packets = []
        
        # Generate background benign traffic
        for client_ip in self.network.config['client_ips']:
            bg_generator = TrafficGenerator(client_ip)
            packets.extend(
                bg_generator.generate_benign_traffic(target_ip, duration, rate=5.0)
            )
        
        # Generate attack traffic based on scenario
        if scenario_name == 'syn_flood':
            attack_packets = self.generator.generate_syn_flood(
                target_ip, attack_duration, rate=1000.0
            )
        elif scenario_name == 'udp_flood':
            attack_packets = self.generator.generate_udp_flood(
                target_ip, attack_duration, rate=1000.0
            )
        elif scenario_name == 'http_flood':
            attack_packets = self.generator.generate_http_flood(
                target_ip, attack_duration, rate=500.0
            )
        elif scenario_name == 'dns_amplification':
            attack_packets = self.generator.generate_dns_amplification(
                target_ip, attack_duration, rate=500.0
            )
        elif scenario_name == 'mixed':
            # Multiple attack types
            attack_packets = []
            attack_packets.extend(self.generator.generate_syn_flood(
                target_ip, attack_duration / 3, rate=500.0
            ))
            attack_packets.extend(self.generator.generate_udp_flood(
                target_ip, attack_duration / 3, rate=500.0
            ))
            attack_packets.extend(self.generator.generate_http_flood(
                target_ip, attack_duration / 3, rate=300.0
            ))
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Adjust attack packet timestamps
        attack_start_time = time.time() + attack_start
        for packet in attack_packets:
            packet.timestamp = attack_start_time + (packet.timestamp - attack_packets[0].timestamp)
        
        packets.extend(attack_packets)
        
        # Sort by timestamp
        packets.sort(key=lambda p: p.timestamp)
        
        # Inject into network
        self.network.inject_traffic(packets)
        
        stats = self.network.get_statistics()
        logger.info(f"Scenario completed. Stats: {stats}")
        
        return stats


def create_test_dataset(
    n_samples: int = 10000,
    attack_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a test dataset using the simulated network.
    
    Args:
        n_samples: Approximate number of samples
        attack_ratio: Ratio of attack traffic
        
    Returns:
        Tuple of (features, labels)
    """
    network = SimulatedNetwork()
    scenario = AttackScenario(network)
    
    # Calculate durations
    benign_duration = (1 - attack_ratio) * n_samples / 50  # ~50 pps benign
    attack_duration = attack_ratio * n_samples / 500  # ~500 pps attack
    
    # Run scenario
    scenario.run_scenario(
        'mixed',
        duration=benign_duration + attack_duration,
        attack_start=benign_duration / 2,
        attack_duration=attack_duration
    )
    
    features = network.get_flow_features()
    labels = network.get_flow_labels()
    
    return features, labels


if __name__ == '__main__':
    # Demo: Create simulated network and run attack scenario
    print("Creating simulated network environment...")
    
    network = SimulatedNetwork()
    scenario = AttackScenario(network)
    
    print("\nRunning SYN flood attack scenario...")
    stats = scenario.run_scenario('syn_flood', duration=30.0, attack_start=10.0, attack_duration=10.0)
    
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
    
    print("\nGenerating features from flows...")
    features = network.get_flow_features()
    labels = network.get_flow_labels()
    
    print(f"Features shape: {features.shape}")
    print(f"Labels distribution: {np.bincount(labels)}")
